//! The coaxial coefficients by the recursions of §3.15 (spike S2, SPIKE_REPORT §2.1).
//!
//! Everything runs on the scaled Êᵐ_{n′n} = Eᵐ_{n′n} g^{n+n′} (g = s for (S|R), 1 for
//! (R|R)), which puts g² on the terms that lower n + n′ and needs the start values
//! f̂ₗ = fₗ gˡ; the exponential factor of the stored normalisation is common to the start
//! values and passes through unchanged. The triangle n′ ≥ n of every order is closed
//! under both recursions (every entry they read is in it), so the rest follows from the
//! index symmetry and costs nothing.

use super::{CoaxialTable, Coefficients};
use crate::radial::{bessel_j, hankel_h};
use crate::{Complex, RealScalar};

/// aᵐₙ = √((n + 1 + m)(n + 1 − m)/((2n + 1)(2n + 3))), zero for n < m.
#[inline]
fn a<T: RealScalar>(n: usize, m: usize) -> T {
    if n < m {
        return T::zero();
    }
    let (n, m) = (n as f64, m as f64);
    T::from_f64((((n + 1.0 + m) * (n + 1.0 - m)) / ((2.0 * n + 1.0) * (2.0 * n + 3.0))).sqrt())
}

/// βₙᵐ = √((n − m − 1)(n − m)/((2n − 1)(2n + 1))), zero for n < m + 2.
#[inline]
fn beta<T: RealScalar>(n: usize, m: usize) -> T {
    if n < m + 2 {
        return T::zero();
    }
    let (n, m) = (n as f64, m as f64);
    T::from_f64((((n - m - 1.0) * (n - m)) / ((2.0 * n - 1.0) * (2.0 * n + 1.0))).sqrt())
}

/// γₙᵐ = √((n + m + 1)(n + m + 2)/((2n + 1)(2n + 3))).
#[inline]
fn gamma<T: RealScalar>(n: usize, m: usize) -> T {
    let (n, m) = (n as f64, m as f64);
    T::from_f64((((n + m + 1.0) * (n + m + 2.0)) / ((2.0 * n + 1.0) * (2.0 * n + 3.0))).sqrt())
}

/// The table of `coefficients` at z with scale g, degrees p_src → p_tgt.
pub(super) fn build<T: RealScalar>(
    coefficients: Coefficients,
    z: Complex<T>,
    g: T,
    p_src: usize,
    p_tgt: usize,
) -> CoaxialTable<T> {
    let mut table = CoaxialTable::zeros(p_src, p_tgt);
    let (rows, cols) = (table.rows, table.cols);
    let l_max = p_src + p_tgt;
    let zero = Complex::new(T::zero(), T::zero());
    // Columns indexed by the row n′, with one slot of padding on each side so that
    // n′ − 1 and n′ + 1 never leave the buffer: row n′ at n′ + 1.
    let len = 2 * rows + 3;
    let mut sect = vec![zero; len];
    match coefficients {
        // Miller's start order grows with the length (`radial`): jₗ is evaluated to the
        // square table's 2 max(p_src, p_tgt), so that a rectangular table is a block of
        // the square one bit for bit. hₗ does not depend on the length.
        Coefficients::Regular => {
            bessel_j(z, T::one(), &mut sect[1..=2 * rows + 1], None);
            sect[l_max + 2..].fill(zero);
        }
        Coefficients::SingularRegular => hankel_h(z, g, &mut sect[1..=l_max + 1], None),
    }
    assert!(
        sect.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
        "the start column of the coaxial table overflows (degrees {p_src} + {p_tgt}, \
         z = ({}, {}), s = {})",
        z.re.to_f64(),
        z.im.to_f64(),
        g.to_f64()
    );
    // E⁰_{n′0} = (−1)^{n′} √(2n′ + 1) f̂_{n′}.
    for n1 in 0..=l_max {
        let root = T::from_f64(((2 * n1 + 1) as f64).sqrt());
        let v = sect[n1 + 1] * root;
        sect[n1 + 1] = if n1 % 2 == 1 { -v } else { v };
    }
    let g2 = g * g;
    let mut tmp = vec![zero; len];
    let mut prev = vec![zero; len];
    let mut cur = vec![zero; len];
    let mut next = vec![zero; len];
    for m in 0..=cols {
        if m > 0 {
            // The sectorial column of order m from that of m − 1, rows m..=l_max − m:
            // γᵐ⁻¹ₘ₋₁ Eᵐ_{n′,m} = βᵐ⁻¹_{n′+1} Eᵐ⁻¹_{n′+1,m−1} + g² γᵐ⁻¹_{n′−1} Eᵐ⁻¹_{n′−1,m−1}.
            let mm = m - 1;
            let inv = T::one() / gamma::<T>(mm, mm);
            tmp.fill(zero);
            for n1 in m..=l_max - m {
                tmp[n1 + 1] = (sect[n1 + 2] * beta::<T>(n1 + 1, mm)
                    + sect[n1] * (g2 * gamma::<T>(n1 - 1, mm)))
                    * inv;
            }
            std::mem::swap(&mut sect, &mut tmp);
        }
        // Column n = m.
        prev.fill(zero);
        cur.fill(zero);
        cur[m + 1..=l_max - m + 1].copy_from_slice(&sect[m + 1..=l_max - m + 1]);
        for (n1, v) in cur.iter().enumerate().take(rows + 2).skip(m + 1) {
            table.set(m, n1 - 1, m, *v);
        }
        // Columns n + 1 = m + 1..=cols, rows n + 1..=l_max − (n + 1), towards the
        // diagonal: aᵐₙ Eᵐ_{n′,n+1} = g² aᵐₙ₋₁ Eᵐ_{n′,n−1} − aᵐ_{n′} Eᵐ_{n′+1,n}
        // + g² aᵐ_{n′−1} Eᵐ_{n′−1,n}.
        for n in m..cols {
            let lo = n + 1;
            let hi = l_max - (n + 1);
            let inv = T::one() / a::<T>(n, m);
            let a_prev = g2 * if n == 0 { T::zero() } else { a::<T>(n - 1, m) };
            next.fill(zero);
            for n1 in lo..=hi {
                let a_lower = if n1 == 0 {
                    T::zero()
                } else {
                    a::<T>(n1 - 1, m)
                };
                next[n1 + 1] = (prev[n1 + 1] * a_prev - cur[n1 + 2] * a::<T>(n1, m)
                    + cur[n1] * (g2 * a_lower))
                    * inv;
            }
            for n1 in lo..=hi.min(rows) {
                table.set(m, n1, n + 1, next[n1 + 1]);
            }
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
    }
    table
}
