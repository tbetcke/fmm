//! Exact relative frames from integer box coordinates (CONVENTIONS §3.13, "Relative
//! frames"; device-path.md §6.1, "Frames from integer coordinates"), shared by the P2P
//! kernels (T6) and the leaf operators (T7).
//!
//! For boxes s and t on levels l_s and l_t with indices i_s and i_t, L = max(l_s, l_t):
//! ĉ(s|t) = ((2 i_s + 1) 2^(L − l_s) − (2 i_t + 1) 2^(L − l_t)) 2^(l_t − L) and
//! r̂(s|t) = 2^(l_t − l_s). The integer difference is formed in `i32` (|N| ≤ 131,070),
//! converted exactly and scaled by an exact power of two, so no step rounds, as
//! `geometry::relative_frame` of `nd-fmm-exec` forms them.

use cubecl::prelude::*;

/// 2⁻¹⁶, exact in f32 and f64.
const TWO_TO_MINUS_16: f32 = 1.0 / 65_536.0;

/// 2ᵉ for −16 ≤ e ≤ 16, exactly: an integer power of two converted, times 2⁻¹⁶ below 1.
#[cube]
pub(crate) fn pow2<F: Float>(e: i32) -> F {
    if e >= 0i32 {
        F::cast_from(1u32 << u32::cast_from(e))
    } else {
        F::cast_from(1u32 << u32::cast_from(e + 16i32)) * F::new(TWO_TO_MINUS_16)
    }
}

/// Component k of the frame centre ĉ(s|t) of box s (level `ls`, index component `is`)
/// seen from box t (level `lt`, index component `it`): exact (module documentation).
#[cube]
pub(crate) fn centre<F: Float>(ls: u32, is: u32, lt: u32, it: u32) -> F {
    let mut big = ls;
    if lt > ls {
        big = lt;
    }
    let cs = (2u32 * is + 1u32) << (big - ls);
    let ct = (2u32 * it + 1u32) << (big - lt);
    let difference = i32::cast_from(cs) - i32::cast_from(ct);
    F::cast_from(difference) * pow2::<F>(i32::cast_from(lt) - i32::cast_from(big))
}

/// The frame ratio r̂(s|t) = 2^(l_t − l_s) of a box on level `ls` seen from one on level
/// `lt`, exact.
#[cube]
pub(crate) fn ratio<F: Float>(ls: u32, lt: u32) -> F {
    pow2::<F>(i32::cast_from(lt) - i32::cast_from(ls))
}
