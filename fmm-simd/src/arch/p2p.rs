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

use super::{MAX_LANES, Simd};
use crate::SimdScalar;

/// P2P on the vector layer `s`: adds into `potential` and, if `G`, into `gradient`
/// the potential and gradient of `sources` with `charges` at `targets` (module
/// documentation), with blocks of `K` vectors.
///
/// `K` is 1, 2 or 4. Without gradients `gradient` is ignored (pass `&mut []`).
///
/// # Panics
///
/// If `sources` and `charges`, `targets` and `potential`, or, with gradients,
/// `targets` and `gradient` differ in length. The public entry point checks these
/// first, with the reference's messages.
#[inline(always)]
pub(crate) fn p2p_body<T: SimdScalar, S: Simd<T>, const K: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    const { assert!(K == 1 || K == 2 || K == 4) };
    const { assert!(S::W <= MAX_LANES) };
    assert_eq!(sources.len(), charges.len(), "`sources` and `charges`");
    assert_eq!(targets.len(), potential.len(), "`targets` and `potential`");
    if G {
        assert_eq!(targets.len(), gradient.len(), "`targets` and `gradient`");
    }
    let w = S::W;
    let n = targets.len();
    let mut start = 0;
    while n - start >= K * w {
        sub_block::<T, S, K, G>(s, sources, charges, targets, potential, gradient, start);
        start += K * w;
    }
    if K >= 4 && n - start >= 2 * w {
        sub_block::<T, S, 2, G>(s, sources, charges, targets, potential, gradient, start);
        start += 2 * w;
    }
    if K >= 2 && n - start >= w {
        sub_block::<T, S, 1, G>(s, sources, charges, targets, potential, gradient, start);
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
        block::<T, S, 1, G>(
            s,
            sources,
            charges,
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

/// [`block`] on the `KB · W` targets from `start` on, in place.
#[inline(always)]
fn sub_block<T: SimdScalar, S: Simd<T>, const KB: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    start: usize,
) {
    let end = start + KB * S::W;
    block::<T, S, KB, G>(
        s,
        sources,
        charges,
        &targets[start..end],
        &mut potential[start..end],
        if G {
            &mut gradient[start..end]
        } else {
            &mut []
        },
    );
}

/// One block of `KB` vectors: `targets` and `potential` (and, if `G`, `gradient`) hold
/// exactly `KB · W` values each.
#[inline(always)]
fn block<T: SimdScalar, S: Simd<T>, const KB: usize, const G: bool>(
    s: S,
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: &mut [[T; 3]],
) {
    let w = S::W;
    let x: [[S::V; 3]; KB] = core::array::from_fn(|k| s.load3(&targets[k * w..]));
    let mut phi: [S::V; KB] = core::array::from_fn(|k| s.load(&potential[k * w..]));
    let zero = s.splat(T::zero());
    let mut g: [[S::V; 3]; KB] = if G {
        core::array::from_fn(|k| s.load3(&gradient[k * w..]))
    } else {
        [[zero; 3]; KB]
    };
    for (y, &q) in sources.iter().zip(charges) {
        let (y, q) = s.broadcast(y, q);
        for k in 0..KB {
            let dx = s.sub(x[k][0], y[0]);
            let dy = s.sub(x[k][1], y[1]);
            let dz = s.sub(x[k][2], y[2]);
            let r2 = s.fma(dz, dz, s.fma(dy, dy, s.mul(dx, dx)));
            let rho = s.rsqrt_masked(r2);
            if G {
                let t = s.mul(q, rho);
                phi[k] = s.add(phi[k], t);
                let t3 = s.mul(t, s.mul(rho, rho));
                g[k][0] = s.fnma(t3, dx, g[k][0]);
                g[k][1] = s.fnma(t3, dy, g[k][1]);
                g[k][2] = s.fnma(t3, dz, g[k][2]);
            } else {
                phi[k] = s.fma(q, rho, phi[k]);
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
