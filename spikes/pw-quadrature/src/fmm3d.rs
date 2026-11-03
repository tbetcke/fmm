//! FMM3D's quadratures of the Helmholtz exponential M2L (FMM3D v2.1.0,
//! `src/Helmholtz/hwts3e.f` and its data files; NIST's, with the notice of those files).
//!
//! FMM3D tabulates them; it does not generate them. `hwts3e(ier, eps, rk, …)` takes the
//! wavenumber in box widths, rk = k w, finds its **domain**, one of 23 × 19 rectangles
//! of complex rk (Re rk on the breakpoints [`RE_BREAKS`], Im rk on [`IM_BREAKS`], up to
//! 16π × 12π), and reads one quadrature per domain and **precision** (`iprec` 0 to 4,
//! from eps: [`precision_of`]), valid uniformly over the rectangle:
//!
//! - `hwts_3d_iprec<p>.txt`: per domain its rectangle, the quadrature type, the error
//!   the file states, and n real nodes xₖ and weights wₖ;
//! - `hnumfour.f`, `hnumphys.f`: per precision and domain, the Fourier counts Fₖ and the
//!   α counts Nₖ of the n nodes.
//!
//! The nodes are mapped to λ by the type ([`Domain::nodes`]): type 1 integrates over real
//! t (λ = √(x² − rk²), weight w x/λ; the domains with Im rk ≥ π/2), type 2 over a contour
//! in λ from −i rk to ∞ (the domains with Im rk < π/2). Type 5 occurs in `hwts3e` but in
//! no table.

use crate::C;
use std::collections::BTreeMap;
use std::path::Path;

/// The breakpoints of Re rk (`hwts3dgetd`'s `cx`, 24 values, 23 intervals).
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

/// The breakpoints of Im rk (`hwts3dgetd`'s `cy`; the first 20 values, 19 intervals,
/// are used).
pub const IM_BREAKS: [f64; 20] = {
    let mut b = [0.0; 20];
    let mut i = 0;
    while i < 20 {
        b[i] = RE_BREAKS[i];
        i += 1;
    }
    b
};

/// Domains along Re rk.
pub const NX: usize = 23;
/// Domains along Im rk.
pub const NY: usize = 19;
/// Domains per precision, `NX * NY`.
pub const DOMAINS: usize = NX * NY;
/// The precisions, `iprec` 0 to 4.
pub const PRECISIONS: usize = 5;

/// `hwts3e`'s precision for a requested eps: 0 (2 digits) for eps ≥ 5·10⁻³, then 1, 2,
/// 3 and 4 below 5·10⁻³, 5·10⁻⁴, 5·10⁻⁷ and 5·10⁻¹⁰.
pub fn precision_of(eps: f64) -> usize {
    [0.5e-2, 0.5e-3, 0.5e-6, 0.5e-9]
        .iter()
        .filter(|&&t| eps < t)
        .count()
}

/// `hwts3e`'s `errmax` of a precision: 10⁻³, 10⁻⁴, 10⁻⁷, 10⁻¹⁰, 10⁻¹³ (set there, never
/// used).
pub fn errmax(precision: usize) -> f64 {
    [1e-3, 1e-4, 1e-7, 1e-10, 1e-13][precision]
}

/// `hwts3dgetd`: the domain (1-based, `ix + (iy − 1) NX`) of rk, or `None` outside the
/// tables (Re rk ≥ 16π or Im rk ≥ 12π). Negative parts map to the first row or column,
/// as there.
pub fn domain_of(rk: C) -> Option<usize> {
    let index = |v: f64, breaks: &[f64], n: usize| {
        let mut i = 0;
        for (j, &b) in breaks.iter().enumerate().take(n + 1).skip(1) {
            if v >= b {
                i = j + 1;
            }
        }
        i.max(1)
    };
    let ix = index(rk.re, &RE_BREAKS, NX);
    let iy = index(rk.im, &IM_BREAKS, NY);
    (ix <= NX && iy <= NY).then_some(ix + (iy - 1) * NX)
}

/// The rectangle of a domain (1-based): its lower and upper corners in rk.
pub fn rectangle(domain: usize) -> (C, C) {
    let ix = (domain - 1) % NX;
    let iy = (domain - 1) / NX;
    (
        C::new(RE_BREAKS[ix], IM_BREAKS[iy]),
        C::new(RE_BREAKS[ix + 1], IM_BREAKS[iy + 1]),
    )
}

/// One tabulated quadrature: a domain at one precision.
#[derive(Clone, Debug)]
pub struct Domain {
    /// The domain, 1-based.
    pub index: usize,
    /// The rectangle's lower corner as the file states it.
    pub rk0: C,
    /// The rectangle's upper corner as the file states it.
    pub rk1: C,
    /// `iquadtype`: 1 or 2 (5 is in `hwts3e` but in no table).
    pub kind: u8,
    /// The error the file states (`err`).
    pub stated_error: f64,
    /// The real nodes xₖ.
    pub xs: Vec<f64>,
    /// The real weights wₖ.
    pub ws: Vec<f64>,
    /// Fₖ, from `hnumfour.f` (empty until the counts are read).
    pub fourier: Vec<usize>,
    /// Nₖ, from `hnumphys.f` (empty until the counts are read).
    pub physical: Vec<usize>,
}

impl Domain {
    /// `hwts3e`'s map of the nodes at rk: the decay rates λₖ and the weights wₖ.
    ///
    /// # Panics
    ///
    /// For a quadrature type other than 1 or 2.
    pub fn nodes(&self, rk: C) -> (Vec<C>, Vec<C>) {
        let i = C::i();
        self.xs
            .iter()
            .zip(&self.ws)
            .map(|(&x, &w)| match self.kind {
                1 => {
                    let lambda = (x * x - rk * rk).sqrt();
                    (lambda, w * x / lambda)
                }
                2 => {
                    // linmap0 to [0, 100], gamma = 0, so v = u.
                    let (u, uweight) = (100.0 * x, 100.0);
                    let a = (-i * rk).im.abs();
                    let lambda = u - i * rk + a * i * (u / (1.0 + u));
                    let weight =
                        (1.0 + a * i * ((1.0 + u) - u) / ((1.0 + u) * (1.0 + u))) * w * uweight;
                    (lambda, weight)
                }
                k => panic!("quadrature type {k} is in no table"),
            })
            .unzip()
    }

    /// The number of nodes, n.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Whether the quadrature has no node.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
}

/// FMM3D's tables, or a subset of them (a fixture): per precision, the domains present.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    /// Per precision (`iprec` 0 to 4), the domains by index.
    pub precisions: [BTreeMap<usize, Domain>; PRECISIONS],
}

impl Tables {
    /// Reads the tables from a directory holding `hwts_3d_iprec<p>.txt` (any of them),
    /// `hnumfour.f` and `hnumphys.f` (FMM3D's `src/Helmholtz`, or a fixture in the same
    /// formats).
    ///
    /// # Panics
    ///
    /// If a file cannot be read or parsed, or a domain's counts do not match its nodes.
    pub fn load(dir: &Path) -> Self {
        let mut tables = Tables::default();
        for p in 0..PRECISIONS {
            let path = dir.join(format!("hwts_3d_iprec{p}.txt"));
            if let Ok(text) = std::fs::read_to_string(&path) {
                tables.precisions[p] = parse_weights(&text);
            }
        }
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let four = parse_counts(&read("hnumfour.f"), "hnumfour", "nfour");
        let phys = parse_counts(&read("hnumphys.f"), "hnumphys", "nphys");
        for (p, domains) in tables.precisions.iter_mut().enumerate() {
            for (index, domain) in domains.iter_mut() {
                let get = |c: &[BTreeMap<usize, Vec<usize>>; PRECISIONS], what: &str| {
                    c[p].get(index)
                        .unwrap_or_else(|| panic!("no {what} for iprec {p}, domain {index}"))
                        .clone()
                };
                domain.fourier = get(&four, "nfour");
                domain.physical = get(&phys, "nphys");
                assert_eq!(domain.fourier.len(), domain.len(), "nfour of {p}/{index}");
                assert_eq!(domain.physical.len(), domain.len(), "nphys of {p}/{index}");
            }
        }
        tables
    }

    /// The quadrature of FMM3D for rk at a precision, as `hwts3e` picks it.
    pub fn lookup(&self, precision: usize, rk: C) -> Option<&Domain> {
        domain_of(rk).and_then(|d| self.precisions[precision].get(&d))
    }
}

/// A Fortran literal (`0.12D+01`) as f64.
fn fortran_f64(s: &str) -> f64 {
    s.trim()
        .replace(['D', 'd'], "e")
        .parse()
        .unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

/// The value after `=` of an assignment line.
fn rhs(line: &str) -> &str {
    line.split_once('=').map(|(_, v)| v.trim()).unwrap_or("")
}

/// The index in `name(  12)` of an assignment's left-hand side.
fn subscript(line: &str) -> usize {
    let open = line.find('(').expect("a subscript");
    let close = line.find(')').expect("a subscript");
    line[open + 1..close].trim().parse().expect("an index")
}

/// The integer of `if( idomain .eq.   30 ) then` or `if(idomain.eq. 30) then`.
fn domain_index(line: &str) -> Option<usize> {
    let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    let rest = compact.strip_prefix("if(idomain.eq.")?;
    rest.split(')').next()?.parse().ok()
}

/// Parses one `hwts_3d_iprec<p>.txt`.
pub fn parse_weights(text: &str) -> BTreeMap<usize, Domain> {
    let mut out = BTreeMap::new();
    let mut cur: Option<Domain> = None;
    let (mut r0, mut r1) = (0.0, 0.0);
    for line in text.lines() {
        if line.starts_with(['c', 'C']) {
            continue;
        }
        let t = line.trim();
        if let Some(index) = domain_index(t) {
            cur = Some(Domain {
                index,
                rk0: C::new(0.0, 0.0),
                rk1: C::new(0.0, 0.0),
                kind: 0,
                stated_error: 0.0,
                xs: Vec::new(),
                ws: Vec::new(),
                fourier: Vec::new(),
                physical: Vec::new(),
            });
            continue;
        }
        let Some(d) = cur.as_mut() else { continue };
        if t == "endif" {
            let d = cur.take().unwrap();
            out.insert(d.index, d);
        } else if t.starts_with("rk0r") {
            r0 = fortran_f64(rhs(t));
        } else if t.starts_with("rk0i") {
            d.rk0 = C::new(r0, fortran_f64(rhs(t)));
        } else if t.starts_with("rk1r") {
            r1 = fortran_f64(rhs(t));
        } else if t.starts_with("rk1i") {
            d.rk1 = C::new(r1, fortran_f64(rhs(t)));
        } else if t.starts_with("iquadtype") {
            d.kind = rhs(t).parse().expect("iquadtype");
        } else if t.starts_with("n ") || t.starts_with("n=") {
            let n: usize = rhs(t).parse().expect("n");
            d.xs.reserve(n);
            d.ws.reserve(n);
        } else if t.starts_with("err") {
            d.stated_error = fortran_f64(rhs(t));
        } else if t.starts_with("xs(") {
            assert_eq!(
                subscript(t),
                d.xs.len() + 1,
                "xs order in domain {}",
                d.index
            );
            d.xs.push(fortran_f64(rhs(t)));
        } else if t.starts_with("ws(") {
            assert_eq!(
                subscript(t),
                d.ws.len() + 1,
                "ws order in domain {}",
                d.index
            );
            d.ws.push(fortran_f64(rhs(t)));
        }
    }
    out
}

/// Parses `hnumfour.f` (`routine` "hnumfour", `array` "nfour") or `hnumphys.f`
/// ("hnumphys", "nphys"): per precision, per domain, the counts.
pub fn parse_counts(
    text: &str,
    routine: &str,
    array: &str,
) -> [BTreeMap<usize, Vec<usize>>; PRECISIONS] {
    let mut out: [BTreeMap<usize, Vec<usize>>; PRECISIONS] = Default::default();
    let mut precision: Option<usize> = None;
    let mut domain: Option<usize> = None;
    let head = format!("subroutine {routine}_iprec");
    for line in text.lines() {
        if line.starts_with(['c', 'C']) {
            continue;
        }
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(&head) {
            precision = rest[..1].parse().ok();
            domain = None;
        } else if t.starts_with("subroutine") {
            precision = None;
        } else if let Some(d) = domain_index(t) {
            domain = Some(d);
        } else if t == "endif" {
            domain = None;
        } else if t.starts_with(array)
            && let (Some(p), Some(d)) = (precision, domain)
        {
            let counts = out[p].entry(d).or_default();
            assert_eq!(subscript(t), counts.len() + 1, "{array} order, {p}/{d}");
            counts.push(rhs(t).parse().expect("a count"));
        }
    }
    out
}
