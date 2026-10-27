//! The coaxial coefficients by the Gumerov–Duraiswami recursions.
//!
//! With Fₙᵐ(r) = fₙ(k r) Yₙᵐ(r̂) and a translation by t = t ẑ,
//! Fₙᵐ(r + t) = Σ_{n′} Eᵐ_{n′n} G_{n′}ᵐ(r). Three identities give every coefficient from
//! the start column E⁰_{n′0} = (−1)^{n′} √(2n′ + 1) f_{n′}(k t):
//!
//! - **in n**, from (1/k) ∂_z Fₙᵐ = aᵐₙ₋₁ Fᵐₙ₋₁ − aᵐₙ Fᵐₙ₊₁ commuting with the translation,
//!   aᵐₙ = √((n + 1 + m)(n + 1 − m)/((2n + 1)(2n + 3))):
//!   aᵐₙ Eᵐ_{n′,n+1} = aᵐₙ₋₁ Eᵐ_{n′,n−1} − aᵐ_{n′} Eᵐ_{n′+1,n} + aᵐ_{n′−1} Eᵐ_{n′−1,n};
//! - **in m**, from (1/k)(∂_x + i∂_y) Fₙᵐ = −βₙᵐ Fᵐ⁺¹ₙ₋₁ − γₙᵐ Fᵐ⁺¹ₙ₊₁ (no
//!   Condon–Shortley phase), βₙᵐ = √((n − m − 1)(n − m)/((2n − 1)(2n + 1))),
//!   γₙᵐ = √((n + m + 1)(n + m + 2)/((2n + 1)(2n + 3))), at n = m:
//!   γᵐₘ Eᵐ⁺¹_{n′,m+1} = βᵐ_{n′+1} Eᵐ_{n′+1,m} + γᵐ_{n′−1} Eᵐ_{n′−1,m};
//! - **the index symmetry** Eᵐ_{nn′} = (−1)^{n+n′} Eᵐ_{n′n}.
//!
//! Everything runs on the scaled Êᵐ_{n′n} = Eᵐ_{n′n} g^{n+n′} (g = 1 for no scaling),
//! which puts g² on the terms that lower n + n′ and needs the start values f̂ₗ = fₗ gˡ.
//! A common factor of the start values (the exponential normalisation) carries through.

use crate::complex::C64;

/// aᵐₙ, zero for n < m.
#[inline]
fn a(n: usize, m: usize) -> f64 {
    if n < m {
        return 0.0;
    }
    let (n, m) = (n as f64, m as f64);
    (((n + 1.0 + m) * (n + 1.0 - m)) / ((2.0 * n + 1.0) * (2.0 * n + 3.0))).sqrt()
}

/// βₙᵐ, zero where (n − m − 1)(n − m) ≤ 0.
#[inline]
fn beta(n: usize, m: usize) -> f64 {
    if n < m + 2 {
        return 0.0;
    }
    let (n, m) = (n as f64, m as f64);
    (((n - m - 1.0) * (n - m)) / ((2.0 * n - 1.0) * (2.0 * n + 1.0))).sqrt()
}

/// γₙᵐ.
#[inline]
fn gamma(n: usize, m: usize) -> f64 {
    let (n, m) = (n as f64, m as f64);
    (((n + m + 1.0) * (n + m + 2.0)) / ((2.0 * n + 1.0) * (2.0 * n + 3.0))).sqrt()
}

/// Which entries the recursion in n computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The lower triangle n′ ≥ n, which the recursions keep closed (every entry it reads
    /// is in it); the rest by the index symmetry. In this triangle the recursion in n
    /// moves toward the diagonal.
    Triangle,
    /// Every row n′ ≥ m of every column, as the recursions are usually written; the
    /// upper triangle n′ < n then comes from marching away from the diagonal.
    Full,
}

/// Runs the recursions for orders 0 ≤ m ≤ min(p_src, p_tgt) and hands each computed
/// column to `sink(m, n, row_lo, values)`, values[i] = Êᵐ_{row_lo + i, n}.
///
/// `start[l]` = f̂ₗ for l = 0..=p_src + p_tgt. Columns n run to p_src (`Full`) or to
/// min(p_src, p_tgt) (`Triangle`); column n holds rows up to p_src + p_tgt − n, more
/// than the table needs (rows up to p_tgt, or max(p_src, p_tgt) for the triangle).
pub fn recurse(
    start: &[C64],
    g: f64,
    p_src: usize,
    p_tgt: usize,
    variant: Variant,
    mut sink: impl FnMut(usize, usize, usize, &[C64]),
) {
    let l_max = p_src + p_tgt;
    assert!(start.len() > l_max, "start needs orders 0..=p_src + p_tgt");
    let g2 = g * g;
    let top_m = p_src.min(p_tgt);
    let last_col = match variant {
        Variant::Triangle => top_m,
        Variant::Full => p_src,
    };
    // Columns indexed by the row n′, with one slot of padding on each side so that
    // n′ − 1 and n′ + 1 never leave the buffer.
    let len = l_max + 3;
    let at = |n1: usize| n1 + 1;
    let mut sect = vec![C64::ZERO; len];
    let mut tmp = vec![C64::ZERO; len];
    let mut prev = vec![C64::ZERO; len];
    let mut cur = vec![C64::ZERO; len];
    let mut next = vec![C64::ZERO; len];
    for (n1, f) in start[..=l_max].iter().enumerate() {
        let sign = if n1 % 2 == 1 { -1.0 } else { 1.0 };
        sect[at(n1)] = *f * (sign * ((2 * n1 + 1) as f64).sqrt());
    }
    for m in 0..=top_m {
        if m > 0 {
            // Sectorial column of order m from that of m − 1 (rows m..=l_max − m).
            let mm = m - 1;
            let inv = 1.0 / gamma(mm, mm);
            tmp.fill(C64::ZERO);
            for n1 in m..=l_max - m {
                tmp[at(n1)] = (sect[at(n1 + 1)] * beta(n1 + 1, mm)
                    + sect[at(n1 - 1)] * (g2 * gamma(n1 - 1, mm)))
                    * inv;
            }
            std::mem::swap(&mut sect, &mut tmp);
        }
        // Column n = m.
        prev.fill(C64::ZERO);
        cur.fill(C64::ZERO);
        cur[at(m)..=at(l_max - m)].copy_from_slice(&sect[at(m)..=at(l_max - m)]);
        sink(m, m, m, &cur[at(m)..=at(l_max - m)]);
        for n in m..last_col {
            let lo = match variant {
                Variant::Triangle => n + 1,
                Variant::Full => m,
            };
            let hi = l_max - (n + 1);
            let inv = 1.0 / a(n, m);
            let a_prev = g2 * if n == 0 { 0.0 } else { a(n - 1, m) };
            next.fill(C64::ZERO);
            for n1 in lo..=hi {
                let a_lower = if n1 == 0 { 0.0 } else { a(n1 - 1, m) };
                next[at(n1)] = (prev[at(n1)] * a_prev - cur[at(n1 + 1)] * a(n1, m)
                    + cur[at(n1) - 1] * (g2 * a_lower))
                    * inv;
            }
            sink(m, n + 1, lo, &next[at(lo)..=at(hi)]);
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
    }
}

/// The halved table: for each order m the triangle n′ ≥ n with rows m..=max(p_src,
/// p_tgt) and columns m..=min(n′, min(p_src, p_tgt)), packed row by row; the other
/// entries by the index symmetry.
pub struct Table {
    /// Largest source degree.
    pub p_src: usize,
    /// Largest target degree.
    pub p_tgt: usize,
    rows: usize,
    cols: usize,
    /// offsets[m * (rows + 1) + n′] is where row n′ of order m starts.
    offsets: Vec<usize>,
    data: Vec<C64>,
    /// degree_max[n′] = max over m and n of |Êᵐ_{n′n}|.
    degree_max: Vec<f64>,
}

impl Table {
    /// Builds the table from the start values (see [`recurse`]) with the triangle
    /// variant.
    pub fn build(start: &[C64], g: f64, p_src: usize, p_tgt: usize) -> Table {
        let rows = p_src.max(p_tgt);
        let cols = p_src.min(p_tgt);
        let mut offsets = vec![0usize; (cols + 1) * (rows + 1)];
        let mut total = 0;
        for m in 0..=cols {
            for n1 in m..=rows {
                offsets[m * (rows + 1) + n1] = total;
                total += n1.min(cols) - m + 1;
            }
        }
        let mut table = Table {
            p_src,
            p_tgt,
            rows,
            cols,
            offsets,
            data: vec![C64::ZERO; total],
            degree_max: vec![0.0; p_tgt + 1],
        };
        recurse(
            start,
            g,
            p_src,
            p_tgt,
            Variant::Triangle,
            |m, n, row_lo, values| {
                for (i, v) in values.iter().enumerate() {
                    let n1 = row_lo + i;
                    if n1 > table.rows {
                        break;
                    }
                    let o = table.offsets[m * (table.rows + 1) + n1] + (n - m);
                    table.data[o] = *v;
                }
            },
        );
        // Each stored (m, r, c), r ≥ c, is Êᵐ_{rc} and, up to sign, Êᵐ_{cr}.
        for m in 0..=cols {
            for r in m..=rows {
                for c in m..=r.min(cols) {
                    let v = table.data[table.offsets[m * (rows + 1) + r] + (c - m)].abs();
                    if r <= p_tgt && c <= p_src {
                        table.degree_max[r] = table.degree_max[r].max(v);
                    }
                    if c <= p_tgt && r <= p_src {
                        table.degree_max[c] = table.degree_max[c].max(v);
                    }
                }
            }
        }
        table
    }

    /// Êᵐ_{n′n} for m ≤ n′ ≤ p_tgt, m ≤ n ≤ p_src.
    pub fn get(&self, m: usize, n1: usize, n: usize) -> C64 {
        debug_assert!(m <= n1.min(n) && n1 <= self.p_tgt && n <= self.p_src);
        if n1 >= n {
            self.data[self.offsets[m * (self.rows + 1) + n1] + (n - m)]
        } else {
            let v = self.data[self.offsets[m * (self.rows + 1) + n] + (n1 - m)];
            if (n + n1) % 2 == 1 { -v } else { v }
        }
    }

    /// Number of stored complex values.
    pub fn stored(&self) -> usize {
        self.data.len()
    }

    /// Largest |Êᵐ_{n′n}| over every order m and source degree n: the size of the
    /// largest contribution to output degree n′. After the rotation of RCR every order
    /// of an input degree carries a share of the same norm, so an error of an entry
    /// matters in proportion to this, not to its own size or to its row of one order.
    pub fn degree_max(&self, n1: usize) -> f64 {
        self.degree_max[n1]
    }

    /// The order count min(p_src, p_tgt) + 1.
    pub fn orders(&self) -> usize {
        self.cols + 1
    }
}
