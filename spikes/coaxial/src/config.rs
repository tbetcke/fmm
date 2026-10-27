//! The configurations of the octree: degrees from proposal §3.5, shifts in units of the
//! circumradius a = √3 w/2 of a box.

/// p of proposal §3.5 (Gumerov–Duraiswami 2009 eq. 18, δ = 2): p = ⌈P⌉ − 1. The same f64
/// arithmetic as `tools/reference.py`.
pub fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// The squared V-list distances in box widths: offsets with largest coordinate 2 or 3.
pub const DISTANCES2: [u32; 15] = [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27];

/// k t of the M2L at squared distance d² (in box widths) on a level with |k| a = ka:
/// t = d w = d · 2a/√3.
pub fn m2l_kt(ka: f64, d2: u32) -> f64 {
    ka * 2.0 * (d2 as f64).sqrt() / 3.0f64.sqrt()
}

/// k t of M2M and L2L between a parent with k a = ka and its child: t = a/2.
pub fn m2m_kt(ka: f64) -> f64 {
    ka / 2.0
}

/// FMM3D's scale at a level with |k| a = ka.
pub fn scale(ka: f64) -> f64 {
    ka.min(1.0)
}
