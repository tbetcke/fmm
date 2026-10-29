//! Rotation blocks of the real orthonormal harmonics at any degree (CONVENTIONS §3.15,
//! "Rotations").
//!
//! The degree-n block Oⁿ(Q) satisfies Yₙ(Q x̂) = Oⁿ(Q) Yₙ(x̂), with Yₙ the 2n + 1
//! harmonics of degree n of [`super::harmonics`]; it equals N Dⁿ(Q) N⁻¹ with Dⁿ the
//! blocks of [`crate::rotation`] (§3.8) and is orthogonal. Coefficients rotate by Oⁿ
//! and back by its transpose.
//!
//! # The y-rotation blocks
//!
//! [`y_blocks`] computes Oⁿ(R_y(β)) by the recursion of Gumerov and Duraiswami (2014),
//! as spike S1 established it (`spikes/rotation-blocks/SPIKE_REPORT.md`; accurate and
//! orthogonal to 10⁻¹⁴ at n = 1000). In the complex orthonormal basis
//! Ŷₙᵐ = νₘ Rₙᵐ, νₘ = √((n + |m|)!(n − |m|)!), the rotation has real entries oₘₖ with
//! Ŷₙᵐ(R_y(β) x) = Σₖ oₘₖ Ŷₙᵏ(x). Only the wedge k ≥ |m| is computed, rows m = −n to n;
//! the rest follows from o₋ₘ,₋ₖ = (−1)^{m+k} oₘₖ and oₖₘ = (−1)^{m+k} oₘₖ. Marching the
//! relation of step 3 over all columns instead is unstable (S1: 4·10¹³³ at n = 500).
//!
//! 1. **Row 0** is the addition theorem of Legendre functions:
//!    o₀ₖ = (−1)ᵏ P̄ₙᵏ(cos β) for k ≥ 0 and P̄ₙ^|k|(cos β) for k < 0.
//! 2. **Row 1** of degree n comes from row 0 of degree n + 1 through the ladder of
//!    CONVENTIONS §3.4 in the rotated frame, with no division by sin β:
//!    𝒟ⁿ₁ₗ = ½(1 − cos β) 𝒟ⁿ⁺¹₀,ₗ₊₁ + ½(1 + cos β) 𝒟ⁿ⁺¹₀,ₗ₋₁ − sin β 𝒟ⁿ⁺¹₀ₗ in the
//!    Racah basis, turned into the orthonormal basis by νₘ.
//! 3. **Rows 2 to n and −1 to −n** by the relation that the rotation commutes with J_y:
//!    with cⱼ = √((n − j)(n + j + 1)),
//!    c_{m′} o_{m′+1,m} = c_{m′−1} o_{m′−1,m} − cₘ o_{m′,m+1} + c_{m−1} o_{m′,m−1},
//!    where o_{m′,n+1} = 0 and cₙ = 0, solved for row m′ + 1 upwards from rows 0 and 1
//!    and for row m′ − 1 downwards, each over k ≥ |row|.
//!
//! In real storage a y-rotation does not mix the cos slots (m ≥ 0) with the sin slots
//! (m < 0). For m, k > 0: O[+m, +k] = oₘₖ + (−1)ᵏ oₘ,₋ₖ, O[−m, −k] = oₘₖ − (−1)ᵏ oₘ,₋ₖ,
//! O[+m, 0] = √2 oₘ₀, O[0, +k] = (o₀ₖ + (−1)ᵏ o₀,₋ₖ)/√2 and O[0, 0] = o₀₀.
//!
//! # Compact layout
//!
//! [`y_blocks`] writes, per degree n from 0 to p, the cos half ((n + 1)², rows and
//! columns m = 0 to n, slots +m) and then the sin half (n², rows and columns m = 1 to n,
//! slots −m), each row-major: [`compact_len`]`(n)` = 2n² + 2n + 1 values at
//! [`compact_offset`]`(n)`, independent of p. [`supplement`] turns the blocks of β into
//! those of π − β in place, by the signs of §3.15. [`blocks`] composes the full blocks
//! of R_z(α) R_y(β) R_z(γ) in the layout of [`crate::rotation::blocks`].

use crate::RealScalar;
use crate::rotation::{block_range, blocks_len};

use super::legendre;

/// Values of the compact block of degree n, (n + 1)² + n².
///
/// ```
/// use nd_fmm_math::orthonormal::rotation::compact_len;
///
/// assert_eq!(compact_len(0), 1);
/// assert_eq!(compact_len(2), 9 + 4);
/// ```
#[inline]
pub const fn compact_len(n: usize) -> usize {
    2 * n * n + 2 * n + 1
}

/// Offset of the compact block of degree n, n(2n² + 1)/3; it does not depend on p.
#[inline]
pub const fn compact_offset(n: usize) -> usize {
    n * (2 * n * n + 1) / 3
}

/// Values of the compact blocks of degrees 0 to p, (p + 1)(2p² + 4p + 3)/3.
#[inline]
pub const fn compact_blocks_len(p: usize) -> usize {
    compact_offset(p + 1)
}

/// Work space of [`y_blocks`] and [`blocks`] up to degree p: the Legendre table to degree
/// p + 1, the wedge of one complex block, the factors cⱼ and one compact block. It
/// allocates once, in [`Scratch::new`].
pub struct Scratch<T> {
    p: usize,
    legendre: Vec<T>,
    wedge: Vec<T>,
    c: Vec<T>,
    compact: Vec<T>,
}

impl<T: RealScalar> Scratch<T> {
    /// Work space for degrees up to `p`.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            legendre: vec![T::zero(); legendre::len(p + 1)],
            wedge: vec![T::zero(); (2 * p + 1) * (p + 1)],
            c: vec![T::zero(); 2 * p + 3],
            compact: vec![T::zero(); compact_len(p)],
        }
    }
}

/// The y-rotation blocks Oⁿ(R_y(β)), n ≤ p, in the compact layout of the
/// [module documentation](self), for x = cos β and s = sin β ≥ 0 (0 ≤ β ≤ π).
///
/// ```
/// use nd_fmm_math::orthonormal::rotation::{Scratch, compact_blocks_len, compact_offset, y_blocks};
///
/// // A quarter turn about y takes z to x: in degree 1, with the cos half over the slots
/// // (0, +1) = (z, x) and the sin half over −1 (y), O¹ = [[0, −1], [1, 0]] and [1].
/// let mut out = vec![0.0; compact_blocks_len(1)];
/// y_blocks(1, 0.0_f64, 1.0, &mut out, &mut Scratch::new(1));
/// let block = &out[compact_offset(1)..compact_offset(2)];
/// let expected = [0.0, -1.0, 1.0, 0.0, 1.0];
/// for (a, e) in block.iter().zip(expected) {
///     assert!((a - e).abs() < 1e-15);
/// }
/// ```
///
/// # Panics
///
/// If `out.len() != compact_blocks_len(p)` or `scratch` is for a smaller degree.
pub fn y_blocks<T: RealScalar>(p: usize, x: T, s: T, out: &mut [T], scratch: &mut Scratch<T>) {
    assert_eq!(
        out.len(),
        compact_blocks_len(p),
        "`out` must have length compact_blocks_len(p) = {} for p = {p}",
        compact_blocks_len(p)
    );
    check_scratch(scratch, p);
    let table = &mut scratch.legendre[..legendre::len(p + 1)];
    legendre::normalised(p + 1, x, s, table);
    for n in 0..=p {
        y_block(
            n,
            table,
            x,
            s,
            &mut scratch.wedge,
            &mut scratch.c,
            &mut out[compact_offset(n)..compact_offset(n + 1)],
        );
    }
}

/// Turns the compact blocks of β, n ≤ p, into those of π − β in place:
/// Oⁿ(R_y(π − β)) = (−1)ⁿ E Oⁿ(R_y(β)) E F, with E = diag((−1)^|m|) and F = +1 on the
/// slots m ≥ 0, −1 on the slots m < 0 (§3.15). The signs are exact.
///
/// # Panics
///
/// If `blocks.len() != compact_blocks_len(p)`.
pub fn supplement<T: RealScalar>(p: usize, blocks: &mut [T]) {
    assert_eq!(
        blocks.len(),
        compact_blocks_len(p),
        "`blocks` must have length compact_blocks_len(p) = {} for p = {p}",
        compact_blocks_len(p)
    );
    for n in 0..=p {
        let block = &mut blocks[compact_offset(n)..compact_offset(n + 1)];
        let (cos, sin) = block.split_at_mut((n + 1) * (n + 1));
        for a in 0..=n {
            for b in 0..=n {
                if (n + a + b) % 2 == 1 {
                    cos[a * (n + 1) + b] = -cos[a * (n + 1) + b];
                }
            }
        }
        for a in 1..=n {
            for b in 1..=n {
                if (n + a + b) % 2 == 0 {
                    sin[(a - 1) * n + (b - 1)] = -sin[(a - 1) * n + (b - 1)];
                }
            }
        }
    }
}

/// The full blocks Oⁿ(Q), n ≤ p, of Q = R_z(α) R_y(β) R_z(γ) (z-y-z, as
/// [`crate::rotation::euler_zyz`]), in the layout of [`crate::rotation::blocks`]: degree
/// by degree, row-major over the slots m = −n to n, entry (i, j) mapping slot j of
/// Yₙ(x̂) to slot i of Yₙ(Q x̂). Composed as Oⁿ(R_z(α)) Oⁿ(R_y(β)) Oⁿ(R_z(γ)), where a
/// z-rotation is a 2 × 2 rotation per order: Yₙ^{±m}(R_z(φ) x̂) mixes the slots ±m by
/// cos mφ and sin mφ. 0 ≤ β ≤ π.
///
/// # Panics
///
/// If `out.len() != blocks_len(p)` or `scratch` is for a smaller degree.
pub fn blocks<T: RealScalar>(
    p: usize,
    alpha: T,
    beta: T,
    gamma: T,
    out: &mut [T],
    scratch: &mut Scratch<T>,
) {
    assert_eq!(
        out.len(),
        blocks_len(p),
        "`out` must have length blocks_len(p) = {} for p = {p}",
        blocks_len(p)
    );
    check_scratch(scratch, p);
    let (s, x) = beta.sin_cos();
    let s = s.max(T::zero());
    let table = &mut scratch.legendre[..legendre::len(p + 1)];
    legendre::normalised(p + 1, x, s, table);
    let (sa, ca) = alpha.sin_cos();
    let (sg, cg) = gamma.sin_cos();
    for n in 0..=p {
        let compact = &mut scratch.compact[..compact_len(n)];
        y_block(n, table, x, s, &mut scratch.wedge, &mut scratch.c, compact);
        let (cos, sin) = compact.split_at((n + 1) * (n + 1));
        // The y-block entry of slots (i, j), zero across the halves.
        let y = |i: isize, j: isize| -> T {
            if i >= 0 && j >= 0 {
                cos[i as usize * (n + 1) + j as usize]
            } else if i < 0 && j < 0 {
                sin[(-i - 1) as usize * n + (-j - 1) as usize]
            } else {
                T::zero()
            }
        };
        let block = &mut out[block_range(n)];
        let w = 2 * n + 1;
        let ni = n as isize;
        // (cos aα, sin aα) for a = |i|, carried across the rows.
        let mut phase_a = (T::one(), T::zero());
        for a in 0..=ni {
            if a > 0 {
                phase_a = rotate(phase_a, (ca, sa));
            }
            let mut phase_b = (T::one(), T::zero());
            for b in 0..=ni {
                if b > 0 {
                    phase_b = rotate(phase_b, (cg, sg));
                }
                for i in [a, -a] {
                    if a == 0 && i < 0 {
                        continue;
                    }
                    for j in [b, -b] {
                        if b == 0 && j < 0 {
                            continue;
                        }
                        block[((i + ni) as usize) * w + (j + ni) as usize] =
                            composed(i, j, a, b, phase_a, phase_b, &y);
                    }
                }
            }
        }
    }
}

/// Entry (i, j) of Z(α) Y Z(γ) for a = |i|, b = |j|. Z(φ) has, for m > 0, row +m with
/// cos mφ at +m and −sin mφ at −m, and row −m with sin mφ at +m and cos mφ at −m, since
/// Rₙᵐ(R_z(φ) x) = e^{imφ} Rₙᵐ(x).
#[inline]
fn composed<T: RealScalar>(
    i: isize,
    j: isize,
    a: isize,
    b: isize,
    (ca, sa): (T, T),
    (cg, sg): (T, T),
    y: &impl Fn(isize, isize) -> T,
) -> T {
    // Row i of Z(α) over the slots +a and −a, and column j of Z(γ) over +b and −b.
    let za = |k: isize| -> T {
        if a == 0 {
            return if k == 0 { T::one() } else { T::zero() };
        }
        match (i > 0, k > 0) {
            (true, true) => ca,
            (true, false) => -sa,
            (false, true) => sa,
            (false, false) => ca,
        }
    };
    let zg = |k: isize| -> T {
        if b == 0 {
            return if k == 0 { T::one() } else { T::zero() };
        }
        match (k > 0, j > 0) {
            (true, true) => cg,
            (true, false) => -sg,
            (false, true) => sg,
            (false, false) => cg,
        }
    };
    let mut sum = za(a) * y(a, b) * zg(b);
    if a > 0 && b > 0 {
        sum = sum + za(-a) * y(-a, -b) * zg(-b);
    }
    sum
}

/// (cos (k + 1)φ, sin (k + 1)φ) from (cos kφ, sin kφ) and (cos φ, sin φ).
#[inline]
fn rotate<T: RealScalar>((c, s): (T, T), (c1, s1): (T, T)) -> (T, T) {
    (c * c1 - s * s1, s * c1 + c * s1)
}

fn check_scratch<T>(scratch: &Scratch<T>, p: usize) {
    assert!(scratch.p >= p, "scratch is for degree {} < {p}", scratch.p);
}

/// (−1)ᵏ.
#[inline(always)]
fn parity<T: RealScalar>(k: isize) -> T {
    if k % 2 == 0 { T::one() } else { -T::one() }
}

/// An integer as `T`.
#[inline(always)]
fn int<T: RealScalar>(n: isize) -> T {
    T::from_f64(n as f64)
}

/// o₀ₖ of degree n from the Legendre table: (−1)ᵏ P̄ₙᵏ for k ≥ 0, P̄ₙ^|k| for k < 0;
/// zero for |k| > n.
#[inline]
fn row_zero<T: RealScalar>(table: &[T], n: usize, k: isize) -> T {
    let a = k.unsigned_abs();
    if a > n {
        return T::zero();
    }
    let value = table[legendre::index(n, a)];
    if k >= 0 {
        parity::<T>(k) * value
    } else {
        value
    }
}

/// (½(1 − x), ½(1 + x)) for x = cos β, s = sin β, the smaller one as s²/(2(1 ± x)), so
/// that neither cancels near β = 0 or β = π.
#[inline]
fn half_one_minus_plus<T: RealScalar>(x: T, s: T) -> (T, T) {
    let half = T::from_f64(0.5);
    if x >= T::zero() {
        (half * s * s / (T::one() + x), half * (T::one() + x))
    } else {
        (half * (T::one() - x), half * s * s / (T::one() - x))
    }
}

/// o₁ₗ of degree n ≥ 1, |l| ≤ n, from row 0 of degree n + 1 (step 2 of the module
/// documentation). With z(k) = o₀ₖ of degree n + 1 and a = |l|,
///
/// o₁ₗ = (½(1 − x) f(l + 1) z(l + 1) + ½(1 + x) f(l − 1) z(l − 1)
///        − s √((n + 1)² − l²) z(l)) / √(n(n + 1)),
///
/// where f(k) = ν^{n+1}_k / νⁿₗ is √((n + a + 1)(n + a + 2)) when |k| = a + 1 and
/// √((n − a + 1)(n − a + 2)) when |k| = a − 1.
#[inline]
fn row_one<T: RealScalar>(table: &[T], n: usize, l: isize, x: T, s: T) -> T {
    let a = l.unsigned_abs() as isize;
    let ni = n as isize;
    let up = int::<T>((ni + a + 1) * (ni + a + 2)).sqrt();
    let down = int::<T>((ni - a + 1) * (ni - a + 2)).sqrt();
    let factor = |k: isize| {
        if k.unsigned_abs() as isize > a {
            up
        } else {
            down
        }
    };
    let z = |k: isize| row_zero(table, n + 1, k);
    let keep = int::<T>((ni + 1) * (ni + 1) - a * a).sqrt();
    let (minus, plus) = half_one_minus_plus(x, s);
    let sum = minus * factor(l + 1) * z(l + 1) + plus * factor(l - 1) * z(l - 1) - s * keep * z(l);
    sum / int::<T>(ni * (ni + 1)).sqrt()
}

/// The compact block of degree n into `out` ([`compact_len`]`(n)` values), from the
/// Legendre table to degree n + 1.
fn y_block<T: RealScalar>(
    n: usize,
    table: &[T],
    x: T,
    s: T,
    wedge: &mut [T],
    c: &mut [T],
    out: &mut [T],
) {
    let ni = n as isize;
    let cols = n + 1;
    // The wedge m ≥ |m′|: rows m′ = −n to n, columns m = 0 to n.
    let wedge = &mut wedge[..(2 * n + 1) * cols];
    let at = |mp: isize, m: isize| ((mp + ni) as usize) * cols + m as usize;
    for m in 0..=ni {
        wedge[at(0, m)] = row_zero(table, n, m);
    }
    for m in 1..=ni {
        wedge[at(1, m)] = row_one(table, n, m, x, s);
    }
    // cⱼ for j = −n − 1 to n, at j + n + 1.
    let c = &mut c[..2 * n + 2];
    for (slot, j) in c.iter_mut().zip(-ni - 1..=ni) {
        *slot = int::<T>((ni - j) * (ni + j + 1)).sqrt();
    }
    let cj = |j: isize| c[(j + ni + 1) as usize];
    // Up, rows 2 to n, each over m ≥ m′ + 1; o_{m′,n+1} is multiplied by cₙ = 0.
    for mp in 1..ni {
        let (inv, lower) = (T::one() / cj(mp), cj(mp - 1));
        for m in mp + 1..=ni {
            let right = if m < ni {
                wedge[at(mp, m + 1)]
            } else {
                T::zero()
            };
            let value =
                lower * wedge[at(mp - 1, m)] - cj(m) * right + cj(m - 1) * wedge[at(mp, m - 1)];
            wedge[at(mp + 1, m)] = value * inv;
        }
    }
    // Down, rows −1 to −n, each over m ≥ 1 − m′:
    // o_{m′−1,m} = (c_{m′} o_{m′+1,m} + cₘ o_{m′,m+1} − c_{m−1} o_{m′,m−1}) / c_{m′−1}.
    for mp in (1 - ni..=0).rev() {
        let (inv, upper) = (T::one() / cj(mp - 1), cj(mp));
        for m in 1 - mp..=ni {
            let right = if m < ni {
                wedge[at(mp, m + 1)]
            } else {
                T::zero()
            };
            let value =
                upper * wedge[at(mp + 1, m)] + cj(m) * right - cj(m - 1) * wedge[at(mp, m - 1)];
            wedge[at(mp - 1, m)] = value * inv;
        }
    }
    // Any entry from the wedge by o_{−m′,−m} = (−1)^{m′+m} o_{m′m} and
    // o_{m m′} = (−1)^{m′+m} o_{m′m}.
    let get = |mp: isize, m: isize| -> T {
        if m >= mp.abs() {
            wedge[at(mp, m)]
        } else if -m >= mp.abs() {
            parity::<T>(mp + m) * wedge[at(-mp, -m)]
        } else if mp > 0 {
            parity::<T>(mp + m) * wedge[at(m, mp)]
        } else {
            wedge[at(-m, -mp)]
        }
    };
    let sqrt2 = T::SQRT_2();
    let (cos, sin) = out.split_at_mut((n + 1) * (n + 1));
    for a in 0..=ni {
        for b in 0..=ni {
            cos[(a as usize) * cols + b as usize] = match (a, b) {
                (0, 0) => get(0, 0),
                (0, _) => (get(0, b) + parity::<T>(b) * get(0, -b)) / sqrt2,
                (_, 0) => sqrt2 * get(a, 0),
                _ => get(a, b) + parity::<T>(b) * get(a, -b),
            };
        }
    }
    for a in 1..=ni {
        for b in 1..=ni {
            sin[((a - 1) * ni + (b - 1)) as usize] = get(a, b) - parity::<T>(b) * get(a, -b);
        }
    }
}
