//! y-rotation blocks of any degree by the Gumerov–Duraiswami (2014) recursion.
//!
//! # The complex blocks
//!
//! In the complex orthonormal basis Ŷₙᵐ = νₘ Rₙᵐ, νₘ = √((n + |m|)! (n − |m|)!), the
//! rotation R_y(β) has real entries oₘₖ with Ŷₙᵐ(R_y(β) x) = Σₖ oₘₖ Ŷₙᵏ(x); in Wigner's
//! standard phase oₘₖ = dⁿₖₘ(β) (tools/reference.py checks this against the definition).
//! Following Gumerov and Duraiswami, only the wedge k ≥ |m| is computed, rows m = −n
//! to n; the rest follows from o₋ₘ,₋ₖ = (−1)^{m+k} oₘₖ and oₖₘ = (−1)^{m+k} oₘₖ. Marching
//! the relation of step 3 over all columns instead is unstable (4·10¹³³ at n = 500 in
//! the spike's first version).
//!
//! 1. **Row 0** is the addition theorem of Legendre functions:
//!    o₀ₖ = (−1)ᵏ P̄ₙᵏ(cos β) for k ≥ 0 and P̄ₙ^|k|(cos β) for k < 0.
//! 2. **Row 1** of degree n comes from row 0 of degree n + 1. Applying
//!    D₊ = ∂₁ + i∂₂ of the rotated frame to Rₙ₊₁⁰(y), y = R_y(β) x, gives −Rₙ¹(y) by the
//!    ladder of CONVENTIONS §3.4; in the frame of x it is the direction
//!    R_y(β)ᵀ(e₁ + i e₂) = (cos β, i, sin β), so in the Racah basis
//!    𝒟ⁿ₁ₗ = ½(1 − cos β) 𝒟ⁿ⁺¹₀,ₗ₊₁ + ½(1 + cos β) 𝒟ⁿ⁺¹₀,ₗ₋₁ − sin β 𝒟ⁿ⁺¹₀ₗ,
//!    with no division by sin β, and νₘ turns it into the orthonormal form of [`row_one`].
//!    This is the second step of Gumerov and Duraiswami.
//! 3. **Rows 2 to n and −1 to −n** by the relation that the rotation commutes with J_y
//!    (their third and fourth steps): with cⱼ = √((n − j)(n + j + 1)),
//!    c_{m′} o_{m′+1,m} = c_{m′−1} o_{m′−1,m} − cₘ o_{m′,m+1} + c_{m−1} o_{m′,m−1},
//!    where o_{m′,n+1} = 0 and cₙ = 0, solved for row m′ + 1 upwards from rows 0 and 1
//!    and for row m′ − 1 downwards from rows 1 and 0, each over k ≥ |row|. It holds in
//!    any basis in which J± have the elements ±cⱼ, so the phase of the basis does not
//!    enter.
//!
//! # The real blocks
//!
//! In real storage a y-rotation does not mix the cos slots (m ≥ 0) with the sin slots
//! (m < 0). For m, k > 0, with Ŷₙ⁻ᵏ = (−1)ᵏ conj(Ŷₙᵏ):
//!
//! - O[+m, +k] = oₘₖ + (−1)ᵏ oₘ,₋ₖ, O[−m, −k] = oₘₖ − (−1)ᵏ oₘ,₋ₖ;
//! - O[+m, 0] = √2 oₘ₀, O[0, +k] = (o₀ₖ + (−1)ᵏ o₀,₋ₖ)/√2, O[0, 0] = o₀₀.
//!
//! # Compact layout
//!
//! [`y_blocks`] writes, per degree n from 0 to p, the cos half ((n + 1)², rows and
//! columns m = 0 to n, slots +m) and then the sin half (n², rows and columns m = 1 to n,
//! slots −m), each row-major: [`compact_len`]`(n)` = 2n² + 2n + 1 values at
//! [`compact_offset`]`(n)`, independent of p.

use crate::entry;
use crate::legendre;

/// Values of the compact block of degree n, (n + 1)² + n².
pub const fn compact_len(n: usize) -> usize {
    2 * n * n + 2 * n + 1
}

/// Offset of the compact block of degree n, Σₖ₍ₖ₌₀..ₙ₋₁₎ (2k² + 2k + 1) =
/// n(2n² + 1)/3.
pub const fn compact_offset(n: usize) -> usize {
    n * (2 * n * n + 1) / 3
}

/// Values of the compact blocks of degrees 0 to p, (p + 1)(2p² + 4p + 3)/3.
pub const fn compact_blocks_len(p: usize) -> usize {
    compact_offset(p + 1)
}

/// Work space of [`y_blocks`] up to degree p: the Legendre table to degree p + 1, the
/// wedge of the complex block and the factors cⱼ.
pub struct Scratch {
    p: usize,
    legendre: Vec<f64>,
    rows: Vec<f64>,
    c: Vec<f64>,
}

impl Scratch {
    /// Work space for degrees up to `p`.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            legendre: vec![0.0; legendre::len(p + 1)],
            rows: vec![0.0; (2 * p + 1) * (p + 1)],
            c: vec![0.0; 2 * p + 3],
        }
    }
}

/// (−1)ᵏ.
#[inline(always)]
fn parity(k: isize) -> f64 {
    if k % 2 == 0 { 1.0 } else { -1.0 }
}

/// o₀ₖ of degree n from the Legendre table: (−1)ᵏ P̄ₙᵏ for k ≥ 0, P̄ₙ^|k| for k < 0;
/// zero for |k| > n.
#[inline]
fn row_zero(legendre: &[f64], n: usize, k: isize) -> f64 {
    let a = k.unsigned_abs();
    if a > n {
        return 0.0;
    }
    let value = legendre[legendre::index(n, a)];
    if k >= 0 { parity(k) * value } else { value }
}

/// (½(1 − x), ½(1 + x)) for x = cos β, s = sin β, the smaller one as s²/(2(1 ± x)), so
/// that neither cancels near β = 0 or β = π (½(1 − cos β) loses all but about
/// −log₁₀(β²/2) of its digits).
#[inline]
fn half_one_minus_plus(x: f64, s: f64) -> (f64, f64) {
    if x >= 0.0 {
        (0.5 * s * s / (1.0 + x), 0.5 * (1.0 + x))
    } else {
        (0.5 * (1.0 - x), 0.5 * s * s / (1.0 - x))
    }
}

/// o₁ₗ of degree n ≥ 1, |l| ≤ n, from row 0 of degree n + 1 (step 2 of the module
/// documentation). With z(k) = o⁰ₖ of degree n + 1 and a = |l|,
///
/// o₁ₗ = (½(1 − x) f(l + 1) z(l + 1) + ½(1 + x) f(l − 1) z(l − 1)
///        − s √((n + 1)² − l²) z(l)) / √(n(n + 1)),
///
/// where f(k) = ν^{n+1}_k / νⁿₗ is √((n + a + 1)(n + a + 2)) when |k| = a + 1 and
/// √((n − a + 1)(n − a + 2)) when |k| = a − 1.
#[inline]
fn row_one(legendre: &[f64], n: usize, l: isize, x: f64, s: f64) -> f64 {
    let a = l.unsigned_abs();
    let nf = n as f64;
    let af = a as f64;
    let up = ((nf + af + 1.0) * (nf + af + 2.0)).sqrt();
    let down = ((nf - af + 1.0) * (nf - af + 2.0)).sqrt();
    let factor = |k: isize| if k.unsigned_abs() > a { up } else { down };
    let z = |k: isize| row_zero(legendre, n + 1, k);
    let keep = ((nf + 1.0) * (nf + 1.0) - af * af).sqrt();
    let (minus, plus) = half_one_minus_plus(x, s);
    let sum = minus * factor(l + 1) * z(l + 1) + plus * factor(l - 1) * z(l - 1) - s * keep * z(l);
    sum / (nf * (nf + 1.0)).sqrt()
}

/// The y-rotation blocks Oⁿ(R_y(β)), n ≤ p, in the compact layout of the
/// [module documentation](self), for x = cos β and s = sin β ≥ 0 (0 ≤ β ≤ π).
///
/// # Panics
///
/// If `out.len() != compact_blocks_len(p)` or `scratch` is for a smaller degree.
pub fn y_blocks(p: usize, x: f64, s: f64, out: &mut [f64], scratch: &mut Scratch) {
    assert_eq!(
        out.len(),
        compact_blocks_len(p),
        "`out` must have length compact_blocks_len(p)"
    );
    assert!(scratch.p >= p, "scratch is for degree {} < {p}", scratch.p);
    let legendre = &mut scratch.legendre[..legendre::len(p + 1)];
    legendre::normalised(p + 1, x, s, legendre);
    let sqrt2 = 2f64.sqrt();
    for n in 0..=p {
        let ni = n as isize;
        let cols = n + 1;
        // The wedge m ≥ |m′|: rows m′ = −n to n, columns m = 0 to n.
        let wedge = &mut scratch.rows[..(2 * n + 1) * cols];
        let at = |mp: isize, m: isize| ((mp + ni) as usize) * cols + m as usize;
        for m in 0..=ni {
            wedge[at(0, m)] = row_zero(legendre, n, m);
        }
        for m in 1..=ni {
            wedge[at(1, m)] = row_one(legendre, n, m, x, s);
        }
        // cⱼ for j = −n − 1 to n, at j + n + 1.
        let c = &mut scratch.c[..2 * n + 2];
        for (slot, j) in c.iter_mut().zip(-ni - 1..=ni) {
            *slot = (((ni - j) * (ni + j + 1)) as f64).sqrt();
        }
        let cj = |j: isize| c[(j + ni + 1) as usize];
        // Up, rows 2 to n, each over m ≥ m′ + 1; o_{m′,n+1} is multiplied by cₙ = 0.
        for mp in 1..ni {
            let (inv, lower) = (1.0 / cj(mp), cj(mp - 1));
            for m in mp + 1..=ni {
                let right = if m < ni { wedge[at(mp, m + 1)] } else { 0.0 };
                let value =
                    lower * wedge[at(mp - 1, m)] - cj(m) * right + cj(m - 1) * wedge[at(mp, m - 1)];
                wedge[at(mp + 1, m)] = value * inv;
            }
        }
        // Down, rows −1 to −n, each over m ≥ 1 − m′:
        // o_{m′−1,m} = (c_{m′} o_{m′+1,m} + cₘ o_{m′,m+1} − c_{m−1} o_{m′,m−1}) / c_{m′−1}.
        for mp in (1 - ni..=0).rev() {
            let (inv, upper) = (1.0 / cj(mp - 1), cj(mp));
            for m in 1 - mp..=ni {
                let right = if m < ni { wedge[at(mp, m + 1)] } else { 0.0 };
                let value =
                    upper * wedge[at(mp + 1, m)] + cj(m) * right - cj(m - 1) * wedge[at(mp, m - 1)];
                wedge[at(mp - 1, m)] = value * inv;
            }
        }
        // Any entry from the wedge by o_{−m′,−m} = (−1)^{m′+m} o_{m′m} and
        // o_{m m′} = (−1)^{m′+m} o_{m′m}.
        let get = |mp: isize, m: isize| -> f64 {
            if m >= mp.abs() {
                wedge[at(mp, m)]
            } else if -m >= mp.abs() {
                parity(mp + m) * wedge[at(-mp, -m)]
            } else if mp > 0 {
                parity(mp + m) * wedge[at(m, mp)]
            } else {
                wedge[at(-m, -mp)]
            }
        };
        // The real block from rows 0 to n.
        let block = &mut out[compact_offset(n)..compact_offset(n + 1)];
        let (cos, sin) = block.split_at_mut((n + 1) * (n + 1));
        for a in 0..=ni {
            for b in 0..=ni {
                cos[(a as usize) * cols + b as usize] = match (a, b) {
                    (0, 0) => get(0, 0),
                    (0, _) => (get(0, b) + parity(b) * get(0, -b)) / sqrt2,
                    (_, 0) => sqrt2 * get(a, 0),
                    _ => get(a, b) + parity(b) * get(a, -b),
                };
            }
        }
        for a in 1..=ni {
            for b in 1..=ni {
                sin[((a - 1) * ni + (b - 1)) as usize] = get(a, b) - parity(b) * get(a, -b);
            }
        }
    }
}

/// Expands the compact block of degree n into the full row-major (2n + 1)² block over
/// the slots −n to n (the layout of [`crate::entry`]), the entries between the halves
/// zero.
pub fn expand(n: usize, compact: &[f64], full: &mut [f64]) {
    assert_eq!(compact.len(), compact_len(n));
    assert_eq!(full.len(), (2 * n + 1) * (2 * n + 1));
    full.fill(0.0);
    let (cos, sin) = compact.split_at((n + 1) * (n + 1));
    for a in 0..=n {
        for b in 0..=n {
            full[entry(n, a as isize, b as isize)] = cos[a * (n + 1) + b];
        }
    }
    for a in 1..=n {
        for b in 1..=n {
            full[entry(n, -(a as isize), -(b as isize))] = sin[(a - 1) * n + (b - 1)];
        }
    }
}

/// The real block of degree n of R_z(φ), slots −n to n: for m > 0, row +m holds
/// cos mφ at +m and −sin mφ at −m, row −m holds sin mφ at +m and cos mφ at −m, since
/// Rₙᵐ(R_z(φ) x) = e^{imφ} Rₙᵐ(x).
pub fn z_block(n: usize, phi: f64, full: &mut [f64]) {
    assert_eq!(full.len(), (2 * n + 1) * (2 * n + 1));
    full.fill(0.0);
    full[entry(n, 0, 0)] = 1.0;
    for m in 1..=n as isize {
        let (sm, cm) = (m as f64 * phi).sin_cos();
        full[entry(n, m, m)] = cm;
        full[entry(n, m, -m)] = -sm;
        full[entry(n, -m, m)] = sm;
        full[entry(n, -m, -m)] = cm;
    }
}

/// Row-major product of two square matrices of width w.
pub fn matmul(w: usize, a: &[f64], b: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..w {
        for k in 0..w {
            let aik = a[i * w + k];
            if aik == 0.0 {
                continue;
            }
            let (row, brow) = (&mut out[i * w..(i + 1) * w], &b[k * w..(k + 1) * w]);
            for (o, &bkj) in row.iter_mut().zip(brow) {
                *o += aik * bkj;
            }
        }
    }
}

/// Full blocks Oⁿ(Q), n ≤ p, of Q = R_z(α) R_y(β) R_z(γ) (z-y-z, as
/// `nd_fmm_math::rotation::euler_zyz`), in the layout of
/// `nd_fmm_math::rotation::blocks`, 0 ≤ β ≤ π. Built as Oⁿ(R_z(α)) Oⁿ(R_y(β)) Oⁿ(R_z(γ)),
/// for comparisons; it allocates.
pub fn euler_blocks(p: usize, alpha: f64, beta: f64, gamma: f64) -> Vec<f64> {
    let (s, x) = beta.sin_cos();
    let mut compact = vec![0.0; compact_blocks_len(p)];
    y_blocks(p, x, s, &mut compact, &mut Scratch::new(p));
    let mut out = Vec::with_capacity(nd_fmm_math::rotation::blocks_len(p));
    for n in 0..=p {
        let w = 2 * n + 1;
        let (mut y, mut za, mut zg, mut t, mut o) = (
            vec![0.0; w * w],
            vec![0.0; w * w],
            vec![0.0; w * w],
            vec![0.0; w * w],
            vec![0.0; w * w],
        );
        expand(
            n,
            &compact[compact_offset(n)..compact_offset(n + 1)],
            &mut y,
        );
        z_block(n, alpha, &mut za);
        z_block(n, gamma, &mut zg);
        matmul(w, &y, &zg, &mut t);
        matmul(w, &za, &t, &mut o);
        out.extend_from_slice(&o);
    }
    out
}
