//! Rotations of real-basis expansions: S1's y-rotation blocks by the Gumerov–Duraiswami
//! (2014) recursion (`spikes/rotation-blocks/src/gd.rs`, ported to either precision),
//! the z rotation, and the rotation that takes a translation onto +z.
//!
//! The blocks satisfy Yₙ(Q x) = Oⁿ(Q) Yₙ(x) (S1's definition; it does not see the factor
//! per degree between S1's basis and [`crate::harmonics`]'). An expansion Σ cᵢ Yᵢ(x) has
//! the coefficients O(Q) c in the frame x′ = Q x. For a shift t = |t| t̂ with
//! t̂ = (sin β cos α, sin β sin α, cos β), Q = R_y(−β) R_z(−α) takes t̂ to +z, and
//! O(Q) = O(R_y(β))ᵀ O(R_z(−α)).
//!
//! The Legendre functions of row 0 come from the plain three-term recursion here, not
//! S1's telescoped form: the spike's polar angles are 0, π or at least 0.32 from both,
//! where the plain recursion loses nothing measurable in double-double (S1 needed the
//! telescoped form near β = 10⁻³).

use crate::complex::Complex;
use crate::harmonics::idx;
use crate::real::Real;

/// Values of the compact block of degree n, (n + 1)² + n².
pub const fn compact_len(n: usize) -> usize {
    2 * n * n + 2 * n + 1
}

/// Offset of the compact block of degree n, n(2n² + 1)/3.
pub const fn compact_offset(n: usize) -> usize {
    n * (2 * n * n + 1) / 3
}

/// S1's P̄ₙᵐ = √((n − m)!/(n + m)!) Pₙᵐ(x), 0 ≤ m ≤ n ≤ nmax, at n(n + 1)/2 + m.
fn legendre<T: Real>(nmax: usize, x: T, s: T) -> Vec<T> {
    let at = |n: usize, m: usize| n * (n + 1) / 2 + m;
    let mut out = vec![T::ZERO; (nmax + 1) * (nmax + 2) / 2];
    let mut diag = T::ONE;
    for m in 0..=nmax {
        if m > 0 {
            diag = diag * (T::from_usize(2 * m - 1) / T::from_usize(2 * m)).sqrt() * s;
        }
        out[at(m, m)] = diag;
        if m == nmax {
            break;
        }
        out[at(m + 1, m)] = T::from_usize(2 * m + 1).sqrt() * x * diag;
        for n in m + 2..=nmax {
            out[at(n, m)] = (T::from_usize(2 * n - 1) * x * out[at(n - 1, m)]
                - T::from_usize((n + m - 1) * (n - m - 1)).sqrt() * out[at(n - 2, m)])
                / T::from_usize((n - m) * (n + m)).sqrt();
        }
    }
    out
}

/// The y-rotation blocks Oⁿ(R_y(β)), n ≤ p, in S1's compact layout: per degree the cos
/// half ((n + 1)², rows and columns m = 0..n, slots +m) then the sin half (n², rows and
/// columns m = 1..n, slots −m), row-major.
pub struct YBlocks<T> {
    /// The largest degree.
    pub p: usize,
    data: Vec<T>,
}

impl<T: Real> YBlocks<T> {
    /// The blocks for x = cos β and s = sin β ≥ 0.
    pub fn new(p: usize, x: T, s: T) -> Self {
        let leg = legendre(p + 1, x, s);
        let lat = |n: usize, m: usize| leg[n * (n + 1) / 2 + m];
        let parity = |k: isize| if k % 2 == 0 { T::ONE } else { -T::ONE };
        // o₀ₖ of degree n: (−1)ᵏ P̄ₙᵏ for k ≥ 0, P̄ₙ^|k| for k < 0.
        let row_zero = |n: usize, k: isize| -> T {
            let a = k.unsigned_abs();
            if a > n {
                T::ZERO
            } else if k >= 0 {
                parity(k) * lat(n, a)
            } else {
                lat(n, a)
            }
        };
        let half = T::from_f64(0.5);
        // ½(1 − x), ½(1 + x), the smaller one as s²/(2(1 ± x)).
        let (minus, plus) = if x >= T::ZERO {
            (half * s * s / (T::ONE + x), half * (T::ONE + x))
        } else {
            (half * (T::ONE - x), half * s * s / (T::ONE - x))
        };
        let row_one = |n: usize, l: isize| -> T {
            let a = l.unsigned_abs();
            let up = T::from_usize((n + a + 1) * (n + a + 2)).sqrt();
            let down = if n + 1 >= a {
                T::from_usize((n - a + 1) * (n - a + 2)).sqrt()
            } else {
                T::ZERO
            };
            let factor = |k: isize| if k.unsigned_abs() > a { up } else { down };
            let z = |k: isize| row_zero(n + 1, k);
            let keep = T::from_usize((n + 1) * (n + 1) - a * a).sqrt();
            let sum = minus * factor(l + 1) * z(l + 1) + plus * factor(l - 1) * z(l - 1)
                - s * keep * z(l);
            sum / T::from_usize(n * (n + 1)).sqrt()
        };
        let sqrt2 = T::from_f64(2.0).sqrt();
        let mut data = vec![T::ZERO; compact_offset(p + 1)];
        let mut wedge = vec![T::ZERO; (2 * p + 1) * (p + 1)];
        let mut c = vec![T::ZERO; 2 * p + 2];
        for n in 0..=p {
            let ni = n as isize;
            let cols = n + 1;
            let at = |mp: isize, m: isize| ((mp + ni) as usize) * cols + m as usize;
            for m in 0..=ni {
                wedge[at(0, m)] = row_zero(n, m);
            }
            for m in 1..=ni {
                wedge[at(1, m)] = row_one(n, m);
            }
            for (slot, j) in c.iter_mut().zip(-ni - 1..=ni) {
                *slot = T::from_usize(((ni - j) * (ni + j + 1)) as usize).sqrt();
            }
            let cj = |j: isize| c[(j + ni + 1) as usize];
            for mp in 1..ni {
                let (inv, lower) = (T::ONE / cj(mp), cj(mp - 1));
                for m in mp + 1..=ni {
                    let right = if m < ni {
                        wedge[at(mp, m + 1)]
                    } else {
                        T::ZERO
                    };
                    let value = lower * wedge[at(mp - 1, m)] - cj(m) * right
                        + cj(m - 1) * wedge[at(mp, m - 1)];
                    wedge[at(mp + 1, m)] = value * inv;
                }
            }
            for mp in (1 - ni..=0).rev() {
                let (inv, upper) = (T::ONE / cj(mp - 1), cj(mp));
                for m in 1 - mp..=ni {
                    let right = if m < ni {
                        wedge[at(mp, m + 1)]
                    } else {
                        T::ZERO
                    };
                    let value = upper * wedge[at(mp + 1, m)] + cj(m) * right
                        - cj(m - 1) * wedge[at(mp, m - 1)];
                    wedge[at(mp - 1, m)] = value * inv;
                }
            }
            let get = |mp: isize, m: isize| -> T {
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
            let block = &mut data[compact_offset(n)..compact_offset(n + 1)];
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
        YBlocks { p, data }
    }

    /// The compact block of degree n.
    pub fn block(&self, n: usize) -> &[T] {
        &self.data[compact_offset(n)..compact_offset(n + 1)]
    }

    /// The entry (row, column) of the full block of degree n, slots −n..n.
    pub fn entry(&self, n: usize, row: isize, col: isize) -> T {
        let b = self.block(n);
        let ni = n as isize;
        if row >= 0 && col >= 0 {
            b[(row * (ni + 1) + col) as usize]
        } else if row < 0 && col < 0 {
            b[(n + 1) * (n + 1) + ((-row - 1) * ni + (-col - 1)) as usize]
        } else {
            T::ZERO
        }
    }

    /// out = Oⁿ c (or (Oⁿ)ᵀ c if `transpose`) for every degree n ≤ p of a complex
    /// expansion (both planes).
    pub fn apply(&self, p: usize, transpose: bool, c: &[Complex<T>], out: &mut [Complex<T>]) {
        assert!(p <= self.p);
        for n in 0..=p {
            let b = self.block(n);
            let w = n + 1;
            let (cos, sin) = b.split_at(w * w);
            for a in 0..=n {
                let mut acc = Complex::ZERO;
                for bb in 0..=n {
                    let e = if transpose {
                        cos[bb * w + a]
                    } else {
                        cos[a * w + bb]
                    };
                    acc += c[idx(n, bb as isize)] * e;
                }
                out[idx(n, a as isize)] = acc;
            }
            for a in 1..=n {
                let mut acc = Complex::ZERO;
                for bb in 1..=n {
                    let e = if transpose {
                        sin[(bb - 1) * n + (a - 1)]
                    } else {
                        sin[(a - 1) * n + (bb - 1)]
                    };
                    acc += c[idx(n, -(bb as isize))] * e;
                }
                out[idx(n, -(a as isize))] = acc;
            }
        }
    }
}

/// out = O(R_z(φ)) c for degrees ≤ p, given cos φ and sin φ: for m > 0,
/// out₊ₘ = cos mφ c₊ₘ − sin mφ c₋ₘ and out₋ₘ = sin mφ c₊ₘ + cos mφ c₋ₘ (S1's `z_block`).
pub fn z_apply<T: Real>(p: usize, cos: T, sin: T, c: &[Complex<T>], out: &mut [Complex<T>]) {
    let (mut cm, mut sm) = (T::ONE, T::ZERO);
    let mut cs = Vec::with_capacity(p + 1);
    cs.push((cm, sm));
    for _ in 1..=p {
        (cm, sm) = (cm * cos - sm * sin, sm * cos + cm * sin);
        cs.push((cm, sm));
    }
    for n in 0..=p {
        out[idx(n, 0)] = c[idx(n, 0)];
        for (m, &(cm, sm)) in cs.iter().enumerate().take(n + 1).skip(1) {
            let (plus, minus) = (c[idx(n, m as isize)], c[idx(n, -(m as isize))]);
            out[idx(n, m as isize)] = plus * cm - minus * sm;
            out[idx(n, -(m as isize))] = plus * sm + minus * cm;
        }
    }
}

/// The direction of a shift: (cos β, sin β) and (cos α, sin α) of t̂.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Direction<T> {
    /// cos β.
    pub cb: T,
    /// sin β ≥ 0.
    pub sb: T,
    /// cos α (1 on the axis).
    pub ca: T,
    /// sin α (0 on the axis).
    pub sa: T,
}

impl<T: Real> Direction<T> {
    /// The direction of t ≠ 0.
    pub fn new(t: [T; 3]) -> Self {
        let rho2 = t[0] * t[0] + t[1] * t[1];
        let rho = rho2.sqrt();
        let r = (rho2 + t[2] * t[2]).sqrt();
        let (ca, sa) = if rho2 == T::ZERO {
            (T::ONE, T::ZERO)
        } else {
            (t[0] / rho, t[1] / rho)
        };
        Direction {
            cb: t[2] / r,
            sb: rho / r,
            ca,
            sa,
        }
    }

    /// Q = R_y(−β) R_z(−α) as a matrix, x′ = Q x, with the right-handed R_y(β) =
    /// [[cos β, 0, sin β], [0, 1, 0], [−sin β, 0, cos β]].
    pub fn matrix(&self) -> [[T; 3]; 3] {
        let (cb, sb, ca, sa) = (self.cb, self.sb, self.ca, self.sa);
        // R_z(−α) = [[ca, sa, 0], [−sa, ca, 0], [0, 0, 1]];
        // R_y(−β) = [[cb, 0, −sb], [0, 1, 0], [sb, 0, cb]].
        [
            [cb * ca, cb * sa, -sb],
            [-sa, ca, T::ZERO],
            [sb * ca, sb * sa, cb],
        ]
    }
}

/// The rotation that takes a shift onto +z, with its blocks to degree p.
pub struct Rotation<T> {
    /// The shift's direction.
    pub dir: Direction<T>,
    blocks: YBlocks<T>,
}

impl<T: Real> Rotation<T> {
    /// The rotation for direction `dir`, to degree p.
    pub fn new(dir: Direction<T>, p: usize) -> Self {
        Rotation {
            dir,
            blocks: YBlocks::new(p, dir.cb, dir.sb),
        }
    }

    /// The largest degree.
    pub fn degree(&self) -> usize {
        self.blocks.p
    }

    /// out = O(Q) c: the coefficients in the frame in which the shift lies on +z. `work`
    /// has the length of c.
    pub fn forward(
        &self,
        p: usize,
        c: &[Complex<T>],
        out: &mut [Complex<T>],
        work: &mut [Complex<T>],
    ) {
        z_apply(p, self.dir.ca, -self.dir.sa, c, work);
        self.blocks.apply(p, true, work, out);
    }

    /// out = O(Q)ᵀ c, the inverse of [`Rotation::forward`].
    pub fn back(
        &self,
        p: usize,
        c: &[Complex<T>],
        out: &mut [Complex<T>],
        work: &mut [Complex<T>],
    ) {
        self.blocks.apply(p, false, c, work);
        z_apply(p, self.dir.ca, self.dir.sa, work, out);
    }
}
