//! The P2P kernel body, written once over the vector layer [`Simd`]
//! (docs/design/simd-p2p.md §4.1, §4.2 and §4.5).
//!
//! **Loop order: targets in lanes** (signed off in Phase 3S T2,
//! spikes/p2p-simd/SPIKE_REPORT.md, "Recommendation", item 1). Each lane holds one
//! target; a block holds K vectors of W targets. For each block the body loads the
//! targets, the potentials and, with gradients, the gradients into registers, then
//! visits every source in input order:
//!
//! ```text
//! broadcast yⱼ and qⱼ
//! for each of the K vectors:
//!     d = x − yⱼ                          3 sub
//!     r² = fma(dz, dz, fma(dy, dy, dx²))   mul, 2 fma
//!     ρ = rsqrt(r²), +0 where r² = 0       the layer's inverse square root, compare, and-not
//!     potential only:  φ = fma(qⱼ, ρ, φ)
//!     with gradients:  t = qⱼ ρ;  φ = φ + t;  t₃ = t · (ρ ρ);  gₖ = fnma(t₃, dₖ, gₖ)
//! ```
//!
//! and stores the block. Each target thus adds its sources in input order, starting
//! from the value in the output, with one instruction sequence per lane that does not
//! depend on the other lanes, on K or on where the block starts. That gives chunk
//! invariance and target-position invariance bit for bit.
//!
//! **Blocks and tails** (design §4.5). Whole blocks of K·W targets are loaded from and
//! stored to the caller's slices in place. The rest, fewer than K·W targets, runs as
//! blocks of K/2, K/4, … vectors while whole vectors remain, and the last partial
//! vector as a block of one vector on a stack copy padded with the last target (zero
//! potential and gradient), of which only the real lanes are written back. A smaller
//! block is the same per-lane sequence with fewer independent chains; there is no
//! scalar tail.
//!
//! **The shared skeleton** (Helmholtz CH2.4b; docs/design/helmholtz-fmm-proposal.md §6.2).
//! [`p2p_skeleton`] holds the blocks, the tails, the loads, the broadcast, d and r² and the
//! stores, and takes the arithmetic after r² from a [`PairKernel`], a type parameter
//! resolved at compile time: [`Laplace`] (zero-sized; the source loop above, operation for
//! operation, so [`p2p_body`] is the skeleton with it) and [`Yukawa`] (e^{−κr}/r, design
//! §4.10). The several-vector and dipole bodies below keep their own loops.

use super::exp::exp;
use super::sincos::sincos;
use super::{MAX_LANES, Simd};
use crate::SimdScalar;

/// P2P on the vector layer `s`: adds into `potential` and, if `G`, into `gradient`
/// the potential and gradient of `sources` with `charges` at `targets` (module
/// documentation), with blocks of `K` vectors: [`p2p_skeleton`] with [`Laplace`], one
/// charge and one output plane.
///
/// `K` is 1, 2 or 4. Without gradients `gradient` is ignored (pass `&mut []`).
///
/// # Panics
///
/// As [`p2p_skeleton`].
#[inline(always)]
pub(crate) fn p2p_body<T: SimdScalar, S: Simd<T>, const K: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    p2p_skeleton::<T, S, Laplace, 1, 1, K, G>(
        s,
        Laplace,
        sources,
        charges.as_chunks::<1>().0,
        targets,
        potential,
        gradient,
    );
}

/// The pair arithmetic of a charge kernel on [`p2p_skeleton`] (Helmholtz CH2.4b): what
/// follows d = x − y and r² for one source and one vector of targets, with `QP` charge
/// planes per source and `OP` output planes per target (1 for a real value, 2 for the real
/// and imaginary parts of a complex one; Helmholtz CH3.4b).
///
/// Implementations are `Copy` values whose method is `#[inline(always)]`, so that the
/// arithmetic inlines into each ISA's `#[target_feature]` entry point (crate `CLAUDE.md`,
/// "Inlining check"). The method must be one instruction sequence per lane that does not
/// depend on the other lanes, which keeps chunk and target-position invariance.
pub(crate) trait PairKernel<T: SimdScalar, S: Simd<T>, const QP: usize, const OP: usize>:
    Copy
{
    /// Adds the term of the source with the broadcast charge planes `q` at the target
    /// vector with d = x − y (`d`) and r² (`r2`) into the output planes `phi` and, if `G`,
    /// `g`. A lane with r² = 0 adds +0.
    fn pair<const G: bool>(
        self,
        s: S,
        d: &[S::V; 3],
        r2: S::V,
        q: &[S::V; QP],
        phi: &mut [S::V; OP],
        g: &mut [[S::V; 3]; OP],
    );
}

/// Laplace's pair arithmetic, q/r and −q d/r³ (CONVENTIONS §3.1; module documentation):
/// ρ = rsqrt_masked(r²); potential only φ = fma(q, ρ, φ); with gradients t = q ρ, φ = φ + t,
/// t₃ = t · (ρ ρ), gₖ = fnma(t₃, dₖ, gₖ). Zero-sized; one charge and one output plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Laplace;

impl<T: SimdScalar, S: Simd<T>> PairKernel<T, S, 1, 1> for Laplace {
    #[inline(always)]
    fn pair<const G: bool>(
        self,
        s: S,
        d: &[S::V; 3],
        r2: S::V,
        q: &[S::V; 1],
        phi: &mut [S::V; 1],
        g: &mut [[S::V; 3]; 1],
    ) {
        let (q, phi, g) = (q[0], &mut phi[0], &mut g[0]);
        let rho = s.rsqrt_masked(r2);
        if G {
            let t = s.mul(q, rho);
            *phi = s.add(*phi, t);
            let t3 = s.mul(t, s.mul(rho, rho));
            g[0] = s.fnma(t3, d[0], g[0]);
            g[1] = s.fnma(t3, d[1], g[1]);
            g[2] = s.fnma(t3, d[2], g[2]);
        } else {
            *phi = s.fma(q, rho, *phi);
        }
    }
}

/// Yukawa's pair arithmetic, q e^{−κr}/r and −q (κr + 1) e^{−κr} d/r³ (CONVENTIONS §3.15,
/// k = iκ; design §4.10; Helmholtz CH2.4b):
///
/// ```text
/// ρ = rsqrt_masked(r²);  r = r² ρ;  e = exp((−κ) r)          arch::exp::exp
/// potential only:  φ = fma(q e, ρ, φ)
/// with gradients:  t = (q e) ρ;  φ = φ + t;  f = fma(κ, r, 1);  t₃ = (t f) · (ρ ρ);
///                  gₖ = fnma(t₃, dₖ, gₖ)
/// ```
///
/// A lane with r² = 0 has ρ = r = 0, e = 1 and adds +0. At κ = 0, q e = q and f = 1
/// exactly, so every step equals [`Laplace`]'s and the result is its bit for bit. κ ≥ 0
/// (the caller checks it), so the exponential's argument is never positive. One charge and
/// one output plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Yukawa<V> {
    /// κ in every lane.
    kappa: V,
    /// −κ in every lane.
    minus_kappa: V,
    /// 1 in every lane.
    one: V,
}

impl<V: Copy> Yukawa<V> {
    /// The pair arithmetic for the wavenumber `kappa` ≥ 0 on the layer `s`.
    #[inline(always)]
    pub(crate) fn new<T: SimdScalar, S: Simd<T, V = V>>(s: S, kappa: T) -> Self {
        Self {
            kappa: s.splat(kappa),
            minus_kappa: s.splat(-kappa),
            one: s.splat(T::one()),
        }
    }
}

impl<T: SimdScalar, S: Simd<T>> PairKernel<T, S, 1, 1> for Yukawa<S::V> {
    #[inline(always)]
    fn pair<const G: bool>(
        self,
        s: S,
        d: &[S::V; 3],
        r2: S::V,
        q: &[S::V; 1],
        phi: &mut [S::V; 1],
        g: &mut [[S::V; 3]; 1],
    ) {
        let (q, phi, g) = (q[0], &mut phi[0], &mut g[0]);
        let rho = s.rsqrt_masked(r2);
        let r = s.mul(r2, rho);
        let e = exp(s, s.mul(self.minus_kappa, r));
        let qe = s.mul(q, e);
        if G {
            let t = s.mul(qe, rho);
            *phi = s.add(*phi, t);
            let f = s.fma(self.kappa, r, self.one);
            let t3 = s.mul(s.mul(t, f), s.mul(rho, rho));
            g[0] = s.fnma(t3, d[0], g[0]);
            g[1] = s.fnma(t3, d[1], g[1]);
            g[2] = s.fnma(t3, d[2], g[2]);
        } else {
            *phi = s.fma(qe, rho, *phi);
        }
    }
}

/// Helmholtz's pair arithmetic, q e^{ikr}/r and q (ikr − 1) e^{ikr} d/r³ for a wavenumber k
/// with Re k ≥ 0 and Im k ≥ 0 (CONVENTIONS §3.15; design §4.12; Helmholtz CH3.4b), with
/// `QP` charge planes (1: real charges q = a; 2: complex charges q = a + i b) and two output
/// planes, the real and the imaginary part:
///
/// ```text
/// ρ = rsqrt_masked(r²);  r = r² ρ;  θ = Re(k) r;  (s, c) = sincos(θ)      arch::sincos
/// DAMPED:   E = exp((−Im k) r);  u_a = a E,  u_b = b E                     arch::exp
/// else:     u_a = a,  u_b = b                                              (Im k = 0)
/// real q:     p_r = u_a c;  p_i = u_a s
/// complex q:  p_r = fnma(u_b, s, u_a c);  p_i = fma(u_b, c, u_a s)         q e^{ikr}
/// potential only:  φ_r = fma(p_r, ρ, φ_r);  φ_i = fma(p_i, ρ, φ_i)
/// with gradients:  t_r = p_r ρ;  t_i = p_i ρ;  φ_r = φ_r + t_r;  φ_i = φ_i + t_i
///                  DAMPED: f = fma(Im k, r, 1);  h_r = fma(t_i, θ, t_r f);  h_i = fnma(t_r, θ, t_i f)
///                  else:   h_r = fma(t_i, θ, t_r);  h_i = fnma(t_r, θ, t_i)
///                  ρ₂ = ρ ρ;  gₖ_r = fnma(h_r ρ₂, dₖ, gₖ_r);  gₖ_i = fnma(h_i ρ₂, dₖ, gₖ_i)
/// ```
///
/// With f = 1 + Im(k) r, ikr − 1 = −(f − iθ), so the gradient term is −t (f − iθ) ρ² dₖ and
/// h = t (f − iθ). A lane with r² = 0 has ρ = r = θ = 0, s = +0, c = 1 and E = 1, and adds
/// zeros. At k = 0 with real charges the real plane takes exactly [`Laplace`]'s steps
/// (c = 1, s = +0, h_r = t_r), and at k = iκ (Re k = 0, DAMPED) exactly [`Yukawa`]'s (u_a = q E,
/// h_r = t_r f): the real plane is then theirs bit for bit. `DAMPED` is false only for
/// Im k = 0; the caller chooses it once per call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Helmholtz<V, const DAMPED: bool> {
    /// Re k in every lane.
    re_k: V,
    /// Im k in every lane.
    im_k: V,
    /// −Im k in every lane.
    minus_im_k: V,
    /// 1 in every lane.
    one: V,
}

impl<V: Copy, const DAMPED: bool> Helmholtz<V, DAMPED> {
    /// The pair arithmetic for the wavenumber k = `re_k` + i `im_k`, both ≥ 0 (and `im_k`
    /// zero unless `DAMPED`), on the layer `s`.
    #[inline(always)]
    pub(crate) fn new<T: SimdScalar, S: Simd<T, V = V>>(s: S, re_k: T, im_k: T) -> Self {
        Self {
            re_k: s.splat(re_k),
            im_k: s.splat(im_k),
            minus_im_k: s.splat(-im_k),
            one: s.splat(T::one()),
        }
    }
}

impl<T: SimdScalar, S: Simd<T>, const QP: usize, const DAMPED: bool> PairKernel<T, S, QP, 2>
    for Helmholtz<S::V, DAMPED>
{
    #[inline(always)]
    fn pair<const G: bool>(
        self,
        s: S,
        d: &[S::V; 3],
        r2: S::V,
        q: &[S::V; QP],
        phi: &mut [S::V; 2],
        g: &mut [[S::V; 3]; 2],
    ) {
        let rho = s.rsqrt_masked(r2);
        let r = s.mul(r2, rho);
        let theta = s.mul(self.re_k, r);
        let (sin, cos) = sincos(s, theta);
        let e = if DAMPED {
            exp(s, s.mul(self.minus_im_k, r))
        } else {
            self.one
        };
        let u: [S::V; QP] = core::array::from_fn(|i| if DAMPED { s.mul(q[i], e) } else { q[i] });
        let (pr, pi) = if QP == 1 {
            (s.mul(u[0], cos), s.mul(u[0], sin))
        } else {
            let ub = u[QP - 1];
            (
                s.fnma(ub, sin, s.mul(u[0], cos)),
                s.fma(ub, cos, s.mul(u[0], sin)),
            )
        };
        if G {
            let tr = s.mul(pr, rho);
            let ti = s.mul(pi, rho);
            phi[0] = s.add(phi[0], tr);
            phi[1] = s.add(phi[1], ti);
            let (hr, hi) = if DAMPED {
                let f = s.fma(self.im_k, r, self.one);
                (
                    s.fma(ti, theta, s.mul(tr, f)),
                    s.fnma(tr, theta, s.mul(ti, f)),
                )
            } else {
                (s.fma(ti, theta, tr), s.fnma(tr, theta, ti))
            };
            let rho2 = s.mul(rho, rho);
            let t3r = s.mul(hr, rho2);
            let t3i = s.mul(hi, rho2);
            for k in 0..3 {
                g[0][k] = s.fnma(t3r, d[k], g[0][k]);
                g[1][k] = s.fnma(t3i, d[k], g[1][k]);
            }
        } else {
            phi[0] = s.fma(pr, rho, phi[0]);
            phi[1] = s.fma(pi, rho, phi[1]);
        }
    }
}

/// The P2P skeleton on the vector layer `s` (module documentation): adds into `potential`
/// and, if `G`, into `gradient` the terms of `sources` with the `QP` charge planes of
/// `charges` (point-major) at `targets` that the pair arithmetic `pair` forms, with blocks
/// of `K` vectors, then `K / 2`, …, 1 and the padded tail. The `OP` output planes lie one
/// after the other: plane o of target i at `potential[o n + i]` (and `gradient`), n the
/// number of targets.
///
/// `K` is 1, 2 or 4; `QP` and `OP` 1 or 2. Without gradients `gradient` is ignored (pass
/// `&mut []`).
///
/// # Panics
///
/// If `sources` and `charges` differ in length, or `potential` (or, with gradients,
/// `gradient`) does not hold `OP` values per target. The public entry point checks these
/// first, with the reference's messages.
#[inline(always)]
pub(crate) fn p2p_skeleton<
    T: SimdScalar,
    S: Simd<T>,
    P: PairKernel<T, S, QP, OP>,
    const QP: usize,
    const OP: usize,
    const K: usize,
    const G: bool,
>(
    s: S,
    pair: P,
    sources: &[[T; 3]],
    charges: &[[T; QP]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    const { assert!(K == 1 || K == 2 || K == 4) };
    const { assert!(QP == 1 || QP == 2) };
    const { assert!(OP == 1 || OP == 2) };
    const { assert!(S::W <= MAX_LANES) };
    assert_eq!(sources.len(), charges.len(), "`sources` and `charges`");
    assert_eq!(
        targets.len() * OP,
        potential.len(),
        "`targets` and `potential`"
    );
    if G {
        assert_eq!(
            targets.len() * OP,
            gradient.len(),
            "`targets` and `gradient`"
        );
    }
    let w = S::W;
    let n = targets.len();
    let mut start = 0;
    while n - start >= K * w {
        sub_block::<T, S, P, QP, OP, K, G>(
            s, pair, sources, charges, targets, potential, gradient, start,
        );
        start += K * w;
    }
    if K >= 4 && n - start >= 2 * w {
        sub_block::<T, S, P, QP, OP, 2, G>(
            s, pair, sources, charges, targets, potential, gradient, start,
        );
        start += 2 * w;
    }
    if K >= 2 && n - start >= w {
        sub_block::<T, S, P, QP, OP, 1, G>(
            s, pair, sources, charges, targets, potential, gradient, start,
        );
        start += w;
    }
    let m = n - start;
    if m > 0 && OP == 1 {
        // One plane: the statements of the skeleton before it had planes (Helmholtz
        // CH2.4b), so that Laplace's and Yukawa's code stays the same instruction for
        // instruction (Helmholtz CH3.4b; `fmm-simd/CLAUDE.md`).
        let last = targets[n - 1];
        let mut padded_targets = [last; MAX_LANES];
        padded_targets[..m].copy_from_slice(&targets[start..]);
        let mut padded_potential = [T::zero(); MAX_LANES];
        padded_potential[..m].copy_from_slice(&potential[start..]);
        let mut padded_gradient = [[T::zero(); 3]; MAX_LANES];
        if G {
            padded_gradient[..m].copy_from_slice(&gradient[start..]);
        }
        block::<T, S, P, QP, OP, 1, G>(
            s,
            pair,
            sources,
            charges,
            &padded_targets[..w],
            &mut padded_potential[..w],
            if G {
                &mut padded_gradient[..w]
            } else {
                &mut []
            },
            MAX_LANES,
        );
        potential[start..].copy_from_slice(&padded_potential[..m]);
        if G {
            gradient[start..].copy_from_slice(&padded_gradient[..m]);
        }
    } else if m > 0 {
        // Several planes: a stack copy of one vector of targets, padded with the last
        // target, and of each plane's values there (zero in the padding), plane o at
        // o · MAX_LANES.
        let last = targets[n - 1];
        let mut padded_targets = [last; MAX_LANES];
        padded_targets[..m].copy_from_slice(&targets[start..]);
        let mut padded_potential = [[T::zero(); MAX_LANES]; OP];
        for (o, padded) in padded_potential.iter_mut().enumerate() {
            padded[..m].copy_from_slice(&potential[o * n + start..(o + 1) * n]);
        }
        let mut padded_gradient = [[[T::zero(); 3]; MAX_LANES]; OP];
        if G {
            for (o, padded) in padded_gradient.iter_mut().enumerate() {
                padded[..m].copy_from_slice(&gradient[o * n + start..(o + 1) * n]);
            }
        }
        block::<T, S, P, QP, OP, 1, G>(
            s,
            pair,
            sources,
            charges,
            &padded_targets[..w],
            &mut padded_potential.as_flattened_mut()[..(OP - 1) * MAX_LANES + w],
            if G {
                &mut padded_gradient.as_flattened_mut()[..(OP - 1) * MAX_LANES + w]
            } else {
                &mut []
            },
            MAX_LANES,
        );
        for (o, padded) in padded_potential.iter().enumerate() {
            potential[o * n + start..(o + 1) * n].copy_from_slice(&padded[..m]);
        }
        if G {
            for (o, padded) in padded_gradient.iter().enumerate() {
                gradient[o * n + start..(o + 1) * n].copy_from_slice(&padded[..m]);
            }
        }
    }
}

/// [`block`] on the `KB · W` targets from `start` on, in place.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the shape of `P2pKernel::evaluate` and the vector layer"
)]
fn sub_block<
    T: SimdScalar,
    S: Simd<T>,
    P: PairKernel<T, S, QP, OP>,
    const QP: usize,
    const OP: usize,
    const KB: usize,
    const G: bool,
>(
    s: S,
    pair: P,
    sources: &[[T; 3]],
    charges: &[[T; QP]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    start: usize,
) {
    let n = targets.len();
    let end = start + KB * S::W;
    // One plane: `start..end`, the range of the skeleton before it had planes, so that
    // Laplace's and Yukawa's code stays the same instruction for instruction.
    let last = if OP == 1 { end } else { (OP - 1) * n + end };
    block::<T, S, P, QP, OP, KB, G>(
        s,
        pair,
        sources,
        charges,
        &targets[start..end],
        &mut potential[start..last],
        if G {
            &mut gradient[start..last]
        } else {
            &mut []
        },
        n,
    );
}

/// One block of `KB` vectors: `targets` holds exactly `KB · W` targets, and plane o of
/// target i of the block lies at `potential[o stride + i]` (and, if `G`, `gradient`).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the shape of `P2pKernel::evaluate` and the vector layer"
)]
fn block<
    T: SimdScalar,
    S: Simd<T>,
    P: PairKernel<T, S, QP, OP>,
    const QP: usize,
    const OP: usize,
    const KB: usize,
    const G: bool,
>(
    s: S,
    pair: P,
    sources: &[[T; 3]],
    charges: &[[T; QP]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    stride: usize,
) {
    let w = S::W;
    let at = |o: usize, k: usize| o * stride + k * w;
    let x: [[S::V; 3]; KB] = core::array::from_fn(|k| s.load3(&targets[k * w..]));
    let mut phi: [[S::V; OP]; KB] =
        core::array::from_fn(|k| core::array::from_fn(|o| s.load(&potential[at(o, k)..])));
    let zero = s.splat(T::zero());
    let mut g: [[[S::V; 3]; OP]; KB] = if G {
        core::array::from_fn(|k| core::array::from_fn(|o| s.load3(&gradient[at(o, k)..])))
    } else {
        [[[zero; 3]; OP]; KB]
    };
    for (y, q) in sources.iter().zip(charges) {
        let (y, q0) = s.broadcast(y, q[0]);
        let q: [S::V; QP] = core::array::from_fn(|i| if i == 0 { q0 } else { s.splat(q[i]) });
        for k in 0..KB {
            let dx = s.sub(x[k][0], y[0]);
            let dy = s.sub(x[k][1], y[1]);
            let dz = s.sub(x[k][2], y[2]);
            let r2 = s.fma(dz, dz, s.fma(dy, dy, s.mul(dx, dx)));
            pair.pair::<G>(s, &[dx, dy, dz], r2, &q, &mut phi[k], &mut g[k]);
        }
    }
    for k in 0..KB {
        for o in 0..OP {
            s.store(phi[k][o], &mut potential[at(o, k)..]);
            if G {
                s.store3(g[k][o], &mut gradient[at(o, k)..]);
            }
        }
    }
}

/// The largest vector group of [`p2p_many_body`], the size of the padded stack copies
/// of its tails.
pub(crate) const MAX_GROUP: usize = 8;

/// P2P of m charge vectors (`P2pKernel::evaluate_many`; Phase 6 T5, C6.3,
/// docs/design/optimisation.md §4.3) on the vector layer `s`: `charges` holds the m
/// charges of each source one after the other (`charges[j m + v]`), `potential` and,
/// if `G`, `gradient` the m vectors' values one vector after the other
/// (`potential[v n + i]` for n targets).
///
/// The vectors run in groups of `VG` (then `VG / 2`, …, 1 for the rest), the targets of
/// a group in blocks of `K` vectors as [`p2p_body`]'s. For each source a block forms d,
/// r² and ρ once per target vector and then, for every vector u of the group, the steps
/// of [`p2p_body`]'s source loop with that vector's charge: φ = fma(q, ρ, φ), or with
/// gradients t = q ρ, φ = φ + t, t₃ = t · (ρ ρ), gₖ = fnma(t₃, dₖ, gₖ). Every (target,
/// vector) so gets exactly the instruction sequence of [`p2p_body`] on that vector's
/// charges (ρ is the same value however often it is formed), and vector v of the result
/// equals [`p2p_body`] of vector v bit for bit, whatever `K`, `VG` and the grouping.
///
/// `K` is 1, 2 or 4, `VG` 1, 2, 4 or 8. Without gradients `gradient` is ignored (pass
/// `&mut []`).
///
/// # Panics
///
/// If `m` is zero, or `charges` does not hold m values per source, `potential` (or, with
/// gradients, `gradient`) m per target. The public entry point checks these first.
#[inline(always)]
pub(crate) fn p2p_many_body<
    T: SimdScalar,
    S: Simd<T>,
    const K: usize,
    const VG: usize,
    const G: bool,
>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    m: usize,
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    const { assert!(K == 1 || K == 2 || K == 4) };
    const { assert!(VG == 1 || VG == 2 || VG == 4 || VG == 8) };
    const { assert!(S::W <= MAX_LANES) };
    assert!(m > 0, "at least one charge vector");
    assert_eq!(sources.len() * m, charges.len(), "`sources` and `charges`");
    assert_eq!(
        targets.len() * m,
        potential.len(),
        "`targets` and `potential`"
    );
    if G {
        assert_eq!(
            targets.len() * m,
            gradient.len(),
            "`targets` and `gradient`"
        );
    }
    let mut v0 = 0;
    while m - v0 >= VG {
        group::<T, S, K, VG, G>(s, sources, charges, m, v0, targets, potential, gradient);
        v0 += VG;
    }
    if VG >= 8 && m - v0 >= 4 {
        group::<T, S, K, 4, G>(s, sources, charges, m, v0, targets, potential, gradient);
        v0 += 4;
    }
    if VG >= 4 && m - v0 >= 2 {
        group::<T, S, K, 2, G>(s, sources, charges, m, v0, targets, potential, gradient);
        v0 += 2;
    }
    if VG >= 2 && m - v0 >= 1 {
        group::<T, S, K, 1, G>(s, sources, charges, m, v0, targets, potential, gradient);
    }
}

/// The vectors `v0 .. v0 + VG` of [`p2p_many_body`]: every target, in blocks of `K`
/// vectors, then `K / 2`, …, 1 and a padded tail, as [`p2p_body`].
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the shape of `P2pKernel::evaluate_many` and the vector layer"
)]
fn group<T: SimdScalar, S: Simd<T>, const K: usize, const VG: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    m: usize,
    v0: usize,
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    let w = S::W;
    let n = targets.len();
    let potential = &mut potential[v0 * n..];
    let gradient = if G { &mut gradient[v0 * n..] } else { &mut [] };
    let charges = Group { charges, m, v0 };
    let mut start = 0;
    while n - start >= K * w {
        let block = &targets[start..start + K * w];
        block_many::<T, S, K, VG, G>(s, sources, charges, block, potential, gradient, n, start);
        start += K * w;
    }
    if K >= 4 && n - start >= 2 * w {
        let block = &targets[start..start + 2 * w];
        block_many::<T, S, 2, VG, G>(s, sources, charges, block, potential, gradient, n, start);
        start += 2 * w;
    }
    if K >= 2 && n - start >= w {
        let block = &targets[start..start + w];
        block_many::<T, S, 1, VG, G>(s, sources, charges, block, potential, gradient, n, start);
        start += w;
    }
    let rest = n - start;
    if rest > 0 {
        // A stack copy of one vector of targets, padded with the last target, and of the
        // group's potentials and gradients there (zero in the padding), vector u at u W.
        let mut padded_targets = [targets[n - 1]; MAX_LANES];
        padded_targets[..rest].copy_from_slice(&targets[start..]);
        let mut padded_potential = [T::zero(); MAX_LANES * MAX_GROUP];
        let mut padded_gradient = [[T::zero(); 3]; MAX_LANES * MAX_GROUP];
        for u in 0..VG {
            padded_potential[u * w..u * w + rest]
                .copy_from_slice(&potential[u * n + start..(u + 1) * n]);
            if G {
                padded_gradient[u * w..u * w + rest]
                    .copy_from_slice(&gradient[u * n + start..(u + 1) * n]);
            }
        }
        block_many::<T, S, 1, VG, G>(
            s,
            sources,
            charges,
            &padded_targets[..w],
            &mut padded_potential[..VG * w],
            if G {
                &mut padded_gradient[..VG * w]
            } else {
                &mut []
            },
            w,
            0,
        );
        for u in 0..VG {
            potential[u * n + start..(u + 1) * n]
                .copy_from_slice(&padded_potential[u * w..u * w + rest]);
            if G {
                gradient[u * n + start..(u + 1) * n]
                    .copy_from_slice(&padded_gradient[u * w..u * w + rest]);
            }
        }
    }
}

/// The charges of a vector group: vector `v0 + u` of source j at `charges[j m + v0 + u]`.
#[derive(Clone, Copy)]
struct Group<'a, T> {
    charges: &'a [T],
    m: usize,
    v0: usize,
}

/// One block of `KB` target vectors for the `VG` vectors of a group: `targets` holds
/// exactly `KB · W` targets, and vector u's values of target `start + i` lie at
/// `potential[u stride + start + i]` (and `gradient`, if `G`).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the shape of `P2pKernel::evaluate_many` and the vector layer"
)]
fn block_many<T: SimdScalar, S: Simd<T>, const KB: usize, const VG: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    group: Group<'_, T>,
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    stride: usize,
    start: usize,
) {
    let w = S::W;
    let at = |u: usize, k: usize| u * stride + start + k * w;
    let x: [[S::V; 3]; KB] = core::array::from_fn(|k| s.load3(&targets[k * w..]));
    let mut phi: [[S::V; KB]; VG] =
        core::array::from_fn(|u| core::array::from_fn(|k| s.load(&potential[at(u, k)..])));
    let zero = s.splat(T::zero());
    let mut g: [[[S::V; 3]; KB]; VG] = if G {
        core::array::from_fn(|u| core::array::from_fn(|k| s.load3(&gradient[at(u, k)..])))
    } else {
        [[[zero; 3]; KB]; VG]
    };
    let Group { charges, m, v0 } = group;
    for (y, q) in sources.iter().zip(charges.chunks_exact(m)) {
        let q = &q[v0..v0 + VG];
        let (y, q0) = s.broadcast(y, q[0]);
        for k in 0..KB {
            let dx = s.sub(x[k][0], y[0]);
            let dy = s.sub(x[k][1], y[1]);
            let dz = s.sub(x[k][2], y[2]);
            let r2 = s.fma(dz, dz, s.fma(dy, dy, s.mul(dx, dx)));
            let rho = s.rsqrt_masked(r2);
            if G {
                let rho2 = s.mul(rho, rho);
                for u in 0..VG {
                    let qu = if u == 0 { q0 } else { s.splat(q[u]) };
                    let t = s.mul(qu, rho);
                    phi[u][k] = s.add(phi[u][k], t);
                    let t3 = s.mul(t, rho2);
                    g[u][k][0] = s.fnma(t3, dx, g[u][k][0]);
                    g[u][k][1] = s.fnma(t3, dy, g[u][k][1]);
                    g[u][k][2] = s.fnma(t3, dz, g[u][k][2]);
                }
            } else {
                for u in 0..VG {
                    let qu = if u == 0 { q0 } else { s.splat(q[u]) };
                    phi[u][k] = s.fma(qu, rho, phi[u][k]);
                }
            }
        }
    }
    for u in 0..VG {
        for k in 0..KB {
            s.store(phi[u][k], &mut potential[at(u, k)..]);
            if G {
                s.store3(g[u][k], &mut gradient[at(u, k)..]);
            }
        }
    }
}

/// Dipole P2P (`P2pKernel::evaluate_dipoles`; Phase 6 T8, C6.4; CONVENTIONS §3.7,
/// "Dipole sources") on the vector layer `s`: adds into `potential` and, if `G`, into
/// `gradient` the potential d · (x − y) / |x − y|³ of `sources` with moments `dipoles`, and
/// its gradient, at `targets`, with blocks of `K` vectors and the tails of [`p2p_body`].
///
/// Each lane holds one target and visits the sources in input order:
///
/// ```text
/// broadcast yⱼ and dⱼ
/// for each of the K vectors:
///     w = x − yⱼ                                 3 sub
///     r² = fma(w_z, w_z, fma(w_y, w_y, w_x²))     mul, 2 fma
///     ρ = rsqrt(r²), +0 where r² = 0
///     ρ² = ρ ρ;  ρ³ = ρ ρ²
///     s = fma(d_z, w_z, fma(d_y, w_y, d_x w_x))    mul, 2 fma
///     potential only:  φ = fma(s, ρ³, φ)
///     with gradients:  φ = φ + s ρ³;  a = (3 s) ρ²;  gₖ = fma(ρ³, fnma(a, wₖ, dₖ), gₖ)
/// ```
///
/// so the gradient term is ρ³ (dₖ − 3 s ρ² wₖ) = dₖ / r³ − 3 (d · w) wₖ / r⁵, formed with
/// |3 s ρ² wₖ| ≤ 3 |d| so that no intermediate exceeds the term's bound 4 |d| / r³ by more
/// than ρ³ (CONVENTIONS §3.13, "Range of the terms in f32"). A pair with r² = 0 has
/// ρ = ρ³ = 0 and adds zeros (dₖ is finite). The per-lane sequence does not depend on the
/// other lanes, on `K` or on where the block starts, so chunk and target-position
/// invariance hold as for [`p2p_body`].
///
/// `K` is 1, 2 or 4. Without gradients `gradient` is ignored (pass `&mut []`).
///
/// # Panics
///
/// If `sources` and `dipoles`, `targets` and `potential`, or, with gradients, `targets`
/// and `gradient` differ in length. The public entry point checks these first.
#[inline(always)]
pub(crate) fn p2p_dipoles_body<T: SimdScalar, S: Simd<T>, const K: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    dipoles: &[[T; 3]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    const { assert!(K == 1 || K == 2 || K == 4) };
    const { assert!(S::W <= MAX_LANES) };
    assert_eq!(sources.len(), dipoles.len(), "`sources` and `dipoles`");
    assert_eq!(targets.len(), potential.len(), "`targets` and `potential`");
    if G {
        assert_eq!(targets.len(), gradient.len(), "`targets` and `gradient`");
    }
    let w = S::W;
    let n = targets.len();
    let mut start = 0;
    while n - start >= K * w {
        sub_block_dipoles::<T, S, K, G>(s, sources, dipoles, targets, potential, gradient, start);
        start += K * w;
    }
    if K >= 4 && n - start >= 2 * w {
        sub_block_dipoles::<T, S, 2, G>(s, sources, dipoles, targets, potential, gradient, start);
        start += 2 * w;
    }
    if K >= 2 && n - start >= w {
        sub_block_dipoles::<T, S, 1, G>(s, sources, dipoles, targets, potential, gradient, start);
        start += w;
    }
    let m = n - start;
    if m > 0 {
        let last = targets[n - 1];
        let mut padded_targets = [last; MAX_LANES];
        padded_targets[..m].copy_from_slice(&targets[start..]);
        let mut padded_potential = [T::zero(); MAX_LANES];
        padded_potential[..m].copy_from_slice(&potential[start..]);
        let mut padded_gradient = [[T::zero(); 3]; MAX_LANES];
        if G {
            padded_gradient[..m].copy_from_slice(&gradient[start..]);
        }
        block_dipoles::<T, S, 1, G>(
            s,
            sources,
            dipoles,
            &padded_targets[..w],
            &mut padded_potential[..w],
            if G {
                &mut padded_gradient[..w]
            } else {
                &mut []
            },
        );
        potential[start..].copy_from_slice(&padded_potential[..m]);
        if G {
            gradient[start..].copy_from_slice(&padded_gradient[..m]);
        }
    }
}

/// [`block_dipoles`] on the `KB · W` targets from `start` on, in place.
#[inline(always)]
fn sub_block_dipoles<T: SimdScalar, S: Simd<T>, const KB: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    dipoles: &[[T; 3]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    start: usize,
) {
    let end = start + KB * S::W;
    block_dipoles::<T, S, KB, G>(
        s,
        sources,
        dipoles,
        &targets[start..end],
        &mut potential[start..end],
        if G {
            &mut gradient[start..end]
        } else {
            &mut []
        },
    );
}

/// One block of `KB` vectors of [`p2p_dipoles_body`]: `targets` and `potential` (and, if
/// `G`, `gradient`) hold exactly `KB · W` values each.
#[inline(always)]
fn block_dipoles<T: SimdScalar, S: Simd<T>, const KB: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    dipoles: &[[T; 3]],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    let w = S::W;
    let x: [[S::V; 3]; KB] = core::array::from_fn(|k| s.load3(&targets[k * w..]));
    let mut phi: [S::V; KB] = core::array::from_fn(|k| s.load(&potential[k * w..]));
    let zero = s.splat(T::zero());
    let three = s.splat(T::from_f64(3.0));
    let mut g: [[S::V; 3]; KB] = if G {
        core::array::from_fn(|k| s.load3(&gradient[k * w..]))
    } else {
        [[zero; 3]; KB]
    };
    for (y, d) in sources.iter().zip(dipoles) {
        let (y, _) = s.broadcast(y, T::zero());
        // The moment's three components in every lane, by the same broadcast.
        let (d, _) = s.broadcast(d, T::zero());
        for k in 0..KB {
            let wx = s.sub(x[k][0], y[0]);
            let wy = s.sub(x[k][1], y[1]);
            let wz = s.sub(x[k][2], y[2]);
            let r2 = s.fma(wz, wz, s.fma(wy, wy, s.mul(wx, wx)));
            let rho = s.rsqrt_masked(r2);
            let rho2 = s.mul(rho, rho);
            let rho3 = s.mul(rho, rho2);
            let dw = s.fma(d[2], wz, s.fma(d[1], wy, s.mul(d[0], wx)));
            if G {
                phi[k] = s.add(phi[k], s.mul(dw, rho3));
                let a = s.mul(s.mul(three, dw), rho2);
                g[k][0] = s.fma(rho3, s.fnma(a, wx, d[0]), g[k][0]);
                g[k][1] = s.fma(rho3, s.fnma(a, wy, d[1]), g[k][1]);
                g[k][2] = s.fma(rho3, s.fnma(a, wz, d[2]), g[k][2]);
            } else {
                phi[k] = s.fma(dw, rho3, phi[k]);
            }
        }
    }
    for k in 0..KB {
        s.store(phi[k], &mut potential[k * w..]);
        if G {
            s.store3(g[k], &mut gradient[k * w..]);
        }
    }
}
