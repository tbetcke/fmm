//! The real orthonormal spherical harmonics of S2's basis, with their angular
//! derivatives, in either precision.
//!
//! Yₙ⁰ = p̄ₙ⁰(cos θ)/√(2π), Yₙ⁺ᵐ = p̄ₙᵐ(cos θ) cos mφ/√π, Yₙ⁻ᵐ = p̄ₙᵐ(cos θ) sin mφ/√π,
//! where p̄ₙᵐ are the associated Legendre functions without the Condon–Shortley phase
//! (CONVENTIONS §3.3), normalised to ∫₋₁¹ (p̄ₙᵐ)² du = 1. So the Yₙᵐ are orthonormal on
//! the sphere, Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ) = (2n + 1)/(4π) Pₙ(x̂ · ŷ) (proposal §3.1), and they are
//! the storage of CONVENTIONS §3.6: index n² + n + m, cos in slot +m, sin in slot −m.
//!
//! The recursion in n per order is S2's (`spikes/coaxial/src/projection.rs`). For the
//! gradient, the columns m ≥ 1 run on q̄ₙᵐ = p̄ₙᵐ / sin θ (the recursion in n is linear and
//! free of sin θ, so the column of q̄ starts from p̄ₘᵐ / sin θ), which is finite on the
//! axis; then
//!
//! - ∂_θ p̄ₙ⁰ = −√(n(n + 1)) p̄ₙ¹,
//! - ∂_θ p̄ₙᵐ = ½ (√((n + m)(n − m + 1)) p̄ₙᵐ⁻¹ − √((n − m)(n + m + 1)) p̄ₙᵐ⁺¹), m ≥ 1,
//! - (1/sin θ) ∂_φ Yₙ^{±m} = ∓ m q̄ₙᵐ (sin mφ or −cos mφ)/√π.

use crate::real::Real;

/// Number of coefficients of degrees 0..=p.
pub const fn len(p: usize) -> usize {
    (p + 1) * (p + 1)
}

/// Index of (n, m), −n ≤ m ≤ n.
#[inline(always)]
pub const fn idx(n: usize, m: isize) -> usize {
    (n * n + n).wrapping_add_signed(m)
}

/// aᵐₙ of u p̄ₙᵐ = aᵐₙ p̄ᵐₙ₊₁ + aᵐₙ₋₁ p̄ᵐₙ₋₁.
#[inline]
fn a<T: Real>(n: usize, m: usize) -> T {
    T::from_usize((n + 1 + m) * (n + 1 - m)).sqrt()
        / T::from_usize((2 * n + 1) * (2 * n + 3)).sqrt()
}

/// The polar description of a vector: |x|, cos θ, sin θ, cos φ, sin φ (φ = 0 on the
/// axis).
#[derive(Clone, Copy, Debug)]
pub struct Polar<T> {
    /// |x|.
    pub r: T,
    /// cos θ.
    pub u: T,
    /// sin θ ≥ 0.
    pub s: T,
    /// cos φ.
    pub cp: T,
    /// sin φ.
    pub sp: T,
}

impl<T: Real> Polar<T> {
    /// The polar description of x ≠ 0.
    pub fn new(x: [T; 3]) -> Self {
        let rho2 = x[0] * x[0] + x[1] * x[1];
        let r = (rho2 + x[2] * x[2]).sqrt();
        let rho = rho2.sqrt();
        let (cp, sp) = if rho2 == T::ZERO {
            (T::ONE, T::ZERO)
        } else {
            (x[0] / rho, x[1] / rho)
        };
        Polar {
            r,
            u: x[2] / r,
            s: rho / r,
            cp,
            sp,
        }
    }

    /// r̂, θ̂ and φ̂ in Cartesian components.
    pub fn frame(&self) -> [[T; 3]; 3] {
        [
            [self.s * self.cp, self.s * self.sp, self.u],
            [self.u * self.cp, self.u * self.sp, -self.s],
            [-self.sp, self.cp, T::ZERO],
        ]
    }
}

/// Work space and results of [`Harmonics::eval`] up to degree p.
pub struct Harmonics<T> {
    p: usize,
    /// p̄ₙᵐ at n(n + 1)/2 + m.
    pbar: Vec<T>,
    /// q̄ₙᵐ = p̄ₙᵐ / sin θ, m ≥ 1, at the same index.
    qbar: Vec<T>,
    cos_m: Vec<T>,
    sin_m: Vec<T>,
    /// aᵐₙ at n(n + 1)/2 + m, n ≤ p + 1.
    a_tab: Vec<T>,
    /// √((2m + 1)/(2m)) at m, and √(2m + 3) at p + 2 + m.
    sect_tab: Vec<T>,
    /// √((n + m)(n − m + 1)) at n(n + 1)/2 + m, and √((n − m)(n + m + 1)) at the same
    /// index of `down_tab`.
    up_tab: Vec<T>,
    down_tab: Vec<T>,
    /// Yₙᵐ at [`idx`].
    pub y: Vec<T>,
    /// ∂_θ Yₙᵐ.
    pub dtheta: Vec<T>,
    /// (1/sin θ) ∂_φ Yₙᵐ.
    pub dphi: Vec<T>,
}

#[inline(always)]
fn tri(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

impl<T: Real> Harmonics<T> {
    /// Work space for degrees 0..=p.
    pub fn new(p: usize) -> Self {
        let t = (p + 2) * (p + 3) / 2;
        let mut a_tab = vec![T::ZERO; t];
        let mut up_tab = vec![T::ZERO; t];
        let mut down_tab = vec![T::ZERO; t];
        for n in 0..=p + 1 {
            for m in 0..=n {
                a_tab[tri(n, m)] = a(n, m);
                up_tab[tri(n, m)] = T::from_usize((n + m) * (n - m + 1)).sqrt();
                down_tab[tri(n, m)] = T::from_usize((n - m) * (n + m + 1)).sqrt();
            }
        }
        let mut sect_tab = vec![T::ZERO; 2 * p + 4];
        for m in 0..=p + 1 {
            if m > 0 {
                sect_tab[m] = (T::from_usize(2 * m + 1) / T::from_usize(2 * m)).sqrt();
            }
            sect_tab[p + 2 + m] = T::from_usize(2 * m + 3).sqrt();
        }
        Harmonics {
            p,
            pbar: vec![T::ZERO; t],
            qbar: vec![T::ZERO; t],
            cos_m: vec![T::ZERO; p + 2],
            sin_m: vec![T::ZERO; p + 2],
            a_tab,
            sect_tab,
            up_tab,
            down_tab,
            y: vec![T::ZERO; len(p)],
            dtheta: vec![T::ZERO; len(p)],
            dphi: vec![T::ZERO; len(p)],
        }
    }

    /// The degree.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// Fills [`Harmonics::y`] and, if `gradient`, [`Harmonics::dtheta`] and
    /// [`Harmonics::dphi`] at the direction of `polar`.
    pub fn eval(&mut self, polar: &Polar<T>, gradient: bool) {
        let p = self.p;
        // One degree more than p for ∂_θ, which reads p̄ₙᵐ⁺¹.
        let top = p + usize::from(gradient);
        let (u, s) = (polar.u, polar.s);
        let inv_sqrt2 = T::ONE / T::from_f64(2.0).sqrt();
        // Sectorial q̄ₘᵐ = p̄ₘᵐ / s, m ≥ 1: p̄₁¹ = s √(3/4), p̄ₘᵐ = p̄ₘ₋₁ᵐ⁻¹ s √((2m + 1)/(2m)).
        let mut sect_q = inv_sqrt2 * (T::from_f64(3.0) / T::from_f64(2.0)).sqrt();
        for m in 0..=top {
            if m == 0 {
                self.pbar[0] = inv_sqrt2;
                self.column(0, top, u, false);
                continue;
            }
            if m > 1 {
                sect_q = sect_q * s * self.sect_tab[m];
            }
            self.qbar[tri(m, m)] = sect_q;
            self.column(m, top, u, true);
            for n in m..=top {
                self.pbar[tri(n, m)] = self.qbar[tri(n, m)] * s;
            }
        }
        // cos mφ, sin mφ.
        self.cos_m[0] = T::ONE;
        self.sin_m[0] = T::ZERO;
        for m in 1..=p {
            let (c, sn) = (self.cos_m[m - 1], self.sin_m[m - 1]);
            self.cos_m[m] = c * polar.cp - sn * polar.sp;
            self.sin_m[m] = sn * polar.cp + c * polar.sp;
        }
        let pi = T::pi();
        let n0 = T::ONE / (T::from_f64(2.0) * pi).sqrt();
        let nm = T::ONE / pi.sqrt();
        for n in 0..=p {
            self.y[idx(n, 0)] = self.pbar[tri(n, 0)] * n0;
            for m in 1..=n {
                let v = self.pbar[tri(n, m)] * nm;
                self.y[idx(n, m as isize)] = v * self.cos_m[m];
                self.y[idx(n, -(m as isize))] = v * self.sin_m[m];
            }
        }
        if !gradient {
            return;
        }
        for n in 0..=p {
            let get = |m: usize| {
                if m > n { T::ZERO } else { self.pbar[tri(n, m)] }
            };
            let d0 = -self.up_tab[tri(n, 1.min(n))] * get(1);
            self.dtheta[idx(n, 0)] = d0 * n0;
            self.dphi[idx(n, 0)] = T::ZERO;
            for m in 1..=n {
                let d = (self.up_tab[tri(n, m)] * get(m - 1)
                    - self.down_tab[tri(n, m)] * get(m + 1))
                    * T::from_f64(0.5)
                    * nm;
                self.dtheta[idx(n, m as isize)] = d * self.cos_m[m];
                self.dtheta[idx(n, -(m as isize))] = d * self.sin_m[m];
                let q = self.qbar[tri(n, m)] * T::from_usize(m) * nm;
                self.dphi[idx(n, m as isize)] = -(q * self.sin_m[m]);
                self.dphi[idx(n, -(m as isize))] = q * self.cos_m[m];
            }
        }
    }

    /// The column of order m to degree `top` in `pbar` (m = 0) or `qbar`, from its
    /// sectorial value.
    fn column(&mut self, m: usize, top: usize, u: T, q: bool) {
        let col = if q { &mut self.qbar } else { &mut self.pbar };
        if m < top {
            col[tri(m + 1, m)] = u * self.sect_tab[self.p + 2 + m] * col[tri(m, m)];
        }
        for n in m + 1..top {
            col[tri(n + 1, m)] = (u * col[tri(n, m)]
                - self.a_tab[tri(n - 1, m)] * col[tri(n - 1, m)])
                / self.a_tab[tri(n, m)];
        }
    }
}
