//! Plane-wave M2L in the basis of CONVENTIONS §3.3 and the scaling of §3.7: the
//! conversions and the level M2L, direct (one diagonal product per V pair) and with
//! merge-and-shift (Greengard and Rokhlin 1997; Cheng, Greengard and Rokhlin 1999;
//! the scheme of FMM3D's `lfmm3d`).
//!
//! # The representation
//!
//! Work in box widths, u = v/2 for the scaled coordinates v of §3.7 (half-width 1). A
//! scaled multipole gives r φ = Σ M̃ₙᵐ Iₙᵐ(v) = Σ M̂ₙᵐ Iₙᵐ(u) with M̂ₙᵐ = 2^{−(n+1)} M̃ₙᵐ
//! (homogeneity, §3.11). For u_z > 0, 1/|u| = (1/2π) ∫∫ e^{−λ u_z + iλ(u_x cos α +
//! u_y sin α)} dα dλ, and the operator Oₙᵐ of §3.11 ("Irregular addition theorem"),
//! Oₙᵐ I₀⁰ = Iₙᵐ with Oₙᵐ = (−(∂x + i∂y))ᵐ (−∂z)ⁿ⁻ᵐ for m ≥ 0, acts on the plane wave
//! as −∂z → λ, −(∂x + i∂y) → −iλ e^{iα}, ∂x − i∂y → iλ e^{−iα}. So
//!
//! Iₙᵐ(u) = (1/2π) ∫∫ λⁿ cₘ(α) e^{…} dα dλ, cₘ = (−i e^{iα})ᵐ, c₋ₘ = (i e^{−iα})ᵐ,
//!
//! and for real charges (M̂⁻ᵐ = (−1)ᵐ conj M̂ᵐ, §3.6) the orders ±m pair to
//! (−i)ᵐ 2 Re(M̂ᵐ e^{imα}). The quadrature of [`crate::quadrature`] gives the waves
//! X(λₖ, αⱼ) = (wₖ/Nₖ') [Aₖ(0) + Σ_{1≤m<Fₖ'} 2 (−i)ᵐ Re(Aₖ(m) e^{imαⱼ})],
//! Aₖ(m) = Σₙ λₖⁿ M̂ₙᵐ, here with Nₖ' even α nodes. For u_z < 0 (down) the same holds
//! with e^{+λ u_z} and −∂z → −λ: a factor (−1)^{n−m} on each term.
//!
//! X(λ, α + π) = conj X(λ, α), and so is every product with a shift below, so half of the
//! α nodes carry the waves: H = Σₖ Nₖ'/2 complex values per direction.
//!
//! # Translation and the local
//!
//! A shift by d (box widths, from the source centre to the target's) multiplies node
//! (k, j) by T = e^{−λ d_z + iλ(d_x cos α + d_y sin α)} (up; e^{+λ d_z …} down). At the
//! target, D_nᵐ = (∂x − i∂y)ᵐ ∂zⁿ⁻ᵐ picks the coefficient of Rₙᵐ (the ladder of §3.4), and
//! on e^{−λ u_z + iλ(…)} gives λⁿ (−1)ⁿ⁻ᵐ (i e^{−iα})ᵐ (up; λⁿ (i e^{−iα})ᵐ down). With
//! the incoming sum V(λₖ, αⱼ) and Gₖ(m) = Σⱼ V e^{−imαⱼ} over all Nₖ' nodes,
//!
//! L̃ₙᵐ += 2^{−n} λₖⁿ (−1)^{n−m} iᵐ Gₖ(m) (up),  2^{−n} λₖⁿ iᵐ Gₖ(m) (down),
//!
//! slot +m the real part and slot −m the imaginary part (§3.6), m ≤ min(n, Fₖ' − 1).
//!
//! # Directions
//!
//! A pair goes to one of six directions ([`Direction::for_pair`]); ±y and ±x are ±z in a
//! rotated frame: the multipole rotates by K Dⁿ(Q) K and the local back by
//! K S Dⁿ(Qᵀ) S⁻¹ K (§3.11, "Rotation of coefficients"), Q in [`FRAMES`].

use nd_fmm_math::rotation::{block_range, blocks, blocks_len, to_irregular};

use crate::geometry::{Direction, Level, Pair, ParentGroups, children};
use crate::quadrature::Quadrature;

/// The rotation of each frame, taking its direction to +z: the identity (±z),
/// R_x(π/2) with (x, y, z) ↦ (x, −z, y) (±y), and R_y(−π/2) with (x, y, z) ↦ (−z, y, x)
/// (±x); [`Direction::rotate`] applies them to offsets.
pub const FRAMES: [[[f64; 3]; 3]; 3] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
];

/// How the level M2L translates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Translation {
    /// One diagonal product per V pair.
    Direct,
    /// Merge-and-shift through the parents.
    MergeShift,
}

/// The precomputed operator of one degree and quadrature.
#[derive(Clone, Debug)]
pub struct PlaneWave {
    p: usize,
    q: Quadrature,
    /// First node of λₖ in the H values of a direction, and the end: `start[s]` = H.
    start: Vec<usize>,
    /// Fₖ' = min(Fₖ, p + 1).
    modes: Vec<usize>,
    /// λₖⁿ 2^{−(n+1)}, k-major.
    mpow: Vec<f64>,
    /// λₖⁿ 2^{−n}, k-major.
    lpow: Vec<f64>,
    /// wₖ/Nₖ' per λ.
    weight: Vec<f64>,
    /// cos mαⱼ and sin mαⱼ for λₖ: at `trig_start[k] + m · hₖ + j`.
    cos: Vec<f64>,
    sin: Vec<f64>,
    trig_start: Vec<usize>,
    /// The multipole rotations K Dⁿ(Q) K of frames 1 and 2, transposed per degree.
    rotate_multipole: [Vec<f64>; 2],
    /// The local rotations K S Dⁿ(Qᵀ) S⁻¹ K back from frames 1 and 2, transposed.
    rotate_local: [Vec<f64>; 2],
}

/// A complex vector of H values as real and imaginary halves.
fn halves(v: &[f64]) -> (&[f64], &[f64]) {
    v.split_at(v.len() / 2)
}

fn halves_mut(v: &mut [f64]) -> (&mut [f64], &mut [f64]) {
    let h = v.len() / 2;
    v.split_at_mut(h)
}

/// acc += x ⊙ t, complex, over H values.
#[inline]
fn shift_add(acc: &mut [f64], x: &[f64], t: &[f64]) {
    let (ar, ai) = halves_mut(acc);
    let (xr, xi) = halves(x);
    let (tr, ti) = halves(t);
    for ((((ar, ai), &xr), &xi), (&tr, &ti)) in ar
        .iter_mut()
        .zip(ai.iter_mut())
        .zip(xr)
        .zip(xi)
        .zip(tr.iter().zip(ti))
    {
        *ar += xr * tr - xi * ti;
        *ai += xr * ti + xi * tr;
    }
}

/// v = v ⊙ t, complex.
#[inline]
fn shift_in_place(v: &mut [f64], t: &[f64]) {
    let (vr, vi) = halves_mut(v);
    let (tr, ti) = halves(t);
    for (((vr, vi), &tr), &ti) in vr.iter_mut().zip(vi.iter_mut()).zip(tr).zip(ti) {
        let (a, b) = (*vr, *vi);
        *vr = a * tr - b * ti;
        *vi = a * ti + b * tr;
    }
}

/// Σ aᵢ bᵢ with four partial sums (so that it vectorises).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0; 4];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            s[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    (s[0] + s[1]) + (s[2] + s[3]) + tail
}

/// Applies per-degree transposed blocks: out_n += Bᵀ-stored block · in_n.
fn apply_rotation(p: usize, blocks_t: &[f64], input: &[f64], out: &mut [f64]) {
    for n in 0..=p {
        let w = 2 * n + 1;
        let b = &blocks_t[block_range(n)];
        let (i0, o) = (n * n, &mut out[n * n..n * n + w]);
        for j in 0..w {
            let x = input[i0 + j];
            for (o, &bij) in o.iter_mut().zip(&b[j * w..(j + 1) * w]) {
                *o += bij * x;
            }
        }
    }
}

/// K B K (or K S B S⁻¹ K) per degree, stored transposed: entry (i, j) of the block at
/// j (2n + 1) + i.
fn transposed_with_k(p: usize, b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; b.len()];
    for n in 0..=p {
        let w = 2 * n + 1;
        let (src, dst) = (&b[block_range(n)], block_range(n));
        for i in 0..w {
            for j in 0..w {
                let k = |idx: usize| if idx < n { -1.0 } else { 1.0 };
                out[dst.start + j * w + i] = k(i) * k(j) * src[i * w + j];
            }
        }
    }
    out
}

fn transpose(q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| q[j][i]))
}

impl PlaneWave {
    /// The operator of degree `p` with quadrature `q`.
    pub fn new(p: usize, q: Quadrature) -> Self {
        let s = q.nodes();
        let even = q.even_alpha_nodes();
        let mut start = vec![0];
        for &n in &even {
            start.push(start.last().unwrap() + n / 2);
        }
        let modes: Vec<usize> = (0..s).map(|k| q.modes(k, p)).collect();
        let (mut mpow, mut lpow) = (Vec::new(), Vec::new());
        for &lambda in &q.lambdas {
            for n in 0..=p {
                let ln = lambda.powi(n as i32);
                mpow.push(ln / 2f64.powi(n as i32 + 1));
                lpow.push(ln / 2f64.powi(n as i32));
            }
        }
        let weight = (0..s).map(|k| q.weights[k] / even[k] as f64).collect();
        let (mut cos, mut sin, mut trig_start) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..s {
            trig_start.push(cos.len());
            let h = even[k] / 2;
            for m in 0..modes[k] {
                for j in 0..h {
                    let a = core::f64::consts::TAU * (m * j) as f64 / even[k] as f64;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        let mut rotate_multipole = [Vec::new(), Vec::new()];
        let mut rotate_local = [Vec::new(), Vec::new()];
        for f in 1..3 {
            let mut b = vec![0.0; blocks_len(p)];
            blocks(p, &FRAMES[f], &mut b);
            rotate_multipole[f - 1] = transposed_with_k(p, &b);
            blocks(p, &transpose(&FRAMES[f]), &mut b);
            to_irregular(p, &mut b);
            rotate_local[f - 1] = transposed_with_k(p, &b);
        }
        Self {
            p,
            q,
            start,
            modes,
            mpow,
            lpow,
            weight,
            cos,
            sin,
            trig_start,
            rotate_multipole,
            rotate_local,
        }
    }

    /// The degree.
    pub fn p(&self) -> usize {
        self.p
    }

    /// The quadrature.
    pub fn quadrature(&self) -> &Quadrature {
        &self.q
    }

    /// H, complex values per direction.
    pub fn h(&self) -> usize {
        *self.start.last().unwrap()
    }

    /// The shift factors of node (k, j) for the offset (a, b, v) in a direction's frame
    /// (lateral a, b and along v, box widths): e^{−λ|v| + iλ(a cos α + b sin α)}, as
    /// 2H values (real half, imaginary half). |v| because up and down both decay.
    pub fn shift_factors(&self, offset: [f64; 3]) -> Vec<f64> {
        let h = self.h();
        let mut t = vec![0.0; 2 * h];
        let even = self.q.even_alpha_nodes();
        for (k, &lambda) in self.q.lambdas.iter().enumerate() {
            for j in 0..even[k] / 2 {
                let alpha = core::f64::consts::TAU * j as f64 / even[k] as f64;
                let decay = (-lambda * offset[2].abs()).exp();
                let phase = lambda * (offset[0] * alpha.cos() + offset[1] * alpha.sin());
                t[self.start[k] + j] = decay * phase.cos();
                t[h + self.start[k] + j] = decay * phase.sin();
            }
        }
        t
    }

    /// The child shift factors: e^{∓λ v + iλ(…)} for a shift by (a, b, v) with v of any
    /// sign (up: e^{−λ v}; down: e^{+λ v}).
    pub fn signed_shift_factors(&self, offset: [f64; 3], up: bool) -> Vec<f64> {
        let mut t = self.shift_factors([offset[0], offset[1], 0.0]);
        let h = self.h();
        let even = self.q.even_alpha_nodes();
        for (k, &lambda) in self.q.lambdas.iter().enumerate() {
            let e = if up {
                -lambda * offset[2]
            } else {
                lambda * offset[2]
            };
            let e = e.exp();
            for j in 0..even[k] / 2 {
                t[self.start[k] + j] *= e;
                t[h + self.start[k] + j] *= e;
            }
        }
        t
    }

    /// The multipole `m` (scaled, real storage) in `frame`, written to `out`.
    fn multipole_in_frame(&self, frame: usize, m: &[f64], out: &mut [f64]) {
        if frame == 0 {
            out.copy_from_slice(m);
        } else {
            out.fill(0.0);
            apply_rotation(self.p, &self.rotate_multipole[frame - 1], m, out);
        }
    }

    /// The up and down waves (2H each) of the multipole `m`, already in the frame.
    pub fn outgoing(&self, m: &[f64], up: &mut [f64], down: &mut [f64]) {
        let p = self.p;
        let h = self.h();
        up.fill(0.0);
        down.fill(0.0);
        for k in 0..self.q.nodes() {
            let pow = &self.mpow[k * (p + 1)..(k + 1) * (p + 1)];
            let (hk, s0, t0) = (
                self.start[k + 1] - self.start[k],
                self.start[k],
                self.trig_start[k],
            );
            let w = self.weight[k];
            for mm in 0..self.modes[k] {
                // Aₖ(m), even and odd n − m apart.
                let (mut er, mut ei, mut or, mut oi) = (0.0, 0.0, 0.0, 0.0);
                for n in mm..=p {
                    let (re, im) = if mm == 0 {
                        (m[n * n + n], 0.0)
                    } else {
                        (m[n * n + n + mm], m[n * n + n - mm])
                    };
                    if (n - mm) % 2 == 0 {
                        er += pow[n] * re;
                        ei += pow[n] * im;
                    } else {
                        or += pow[n] * re;
                        oi += pow[n] * im;
                    }
                }
                let cos = &self.cos[t0 + mm * hk..t0 + (mm + 1) * hk];
                let sin = &self.sin[t0 + mm * hk..t0 + (mm + 1) * hk];
                for (dir, (ar, ai)) in [
                    (&mut *up, (er + or, ei + oi)),
                    (&mut *down, (er - or, ei - oi)),
                ] {
                    let (wr, wi) = halves_mut(dir);
                    let (wr, wi) = (&mut wr[s0..s0 + hk], &mut wi[s0..s0 + hk]);
                    if mm == 0 {
                        for x in wr.iter_mut() {
                            *x += w * ar;
                        }
                        continue;
                    }
                    // 2 (−i)ᵐ (a cos − b sin) w: into the real part for even m, the
                    // imaginary part for odd m, with the sign of (−i)ᵐ.
                    let sign = if mm % 4 < 2 { 2.0 } else { -2.0 };
                    let (a, b) = if mm % 2 == 0 {
                        (sign * w * ar, -sign * w * ai)
                    } else {
                        (-sign * w * ar, sign * w * ai)
                    };
                    let target = if mm % 2 == 0 { wr } else { wi };
                    for ((x, &c), &s) in target.iter_mut().zip(cos).zip(sin) {
                        *x += a * c + b * s;
                    }
                }
            }
        }
        debug_assert_eq!(up.len(), 2 * h);
    }

    /// Adds the local (scaled, real storage, in the frame) of the incoming up and down
    /// sums `up`, `down` (2H each) to `local`.
    pub fn incoming(&self, up: &[f64], down: &[f64], local: &mut [f64]) {
        let p = self.p;
        for k in 0..self.q.nodes() {
            let pow = &self.lpow[k * (p + 1)..(k + 1) * (p + 1)];
            let (hk, s0, t0) = (
                self.start[k + 1] - self.start[k],
                self.start[k],
                self.trig_start[k],
            );
            let (ur, ui) = halves(up);
            let (dr, di) = halves(down);
            let (ur, ui, dr, di) = (
                &ur[s0..s0 + hk],
                &ui[s0..s0 + hk],
                &dr[s0..s0 + hk],
                &di[s0..s0 + hk],
            );
            for mm in 0..self.modes[k] {
                let cos = &self.cos[t0 + mm * hk..t0 + (mm + 1) * hk];
                let sin = &self.sin[t0 + mm * hk..t0 + (mm + 1) * hk];
                // G(m) over all Nₖ' nodes from half of them (module documentation).
                let g = |re: &[f64], im: &[f64]| {
                    if mm % 2 == 0 {
                        (2.0 * dot(re, cos), -2.0 * dot(re, sin))
                    } else {
                        (2.0 * dot(im, sin), 2.0 * dot(im, cos))
                    }
                };
                let (gu, gd) = (g(ur, ui), g(dr, di));
                // n − m even: up + down; odd: down − up; then times iᵐ.
                let (se, so) = ((gu.0 + gd.0, gu.1 + gd.1), (gd.0 - gu.0, gd.1 - gu.1));
                let rot = |(a, b): (f64, f64)| match mm % 4 {
                    0 => (a, b),
                    1 => (-b, a),
                    2 => (-a, -b),
                    _ => (b, -a),
                };
                let (se, so) = (rot(se), rot(so));
                for n in mm..=p {
                    let (re, im) = if (n - mm) % 2 == 0 { se } else { so };
                    if mm == 0 {
                        local[n * n + n] += pow[n] * re;
                    } else {
                        local[n * n + n + mm] += pow[n] * re;
                        local[n * n + n - mm] += pow[n] * im;
                    }
                }
            }
        }
    }

    /// Adds `local`, in `frame`, rotated back to the original frame, to `out`.
    fn local_from_frame(&self, frame: usize, local: &[f64], out: &mut [f64]) {
        if frame == 0 {
            for (o, &l) in out.iter_mut().zip(local) {
                *o += l;
            }
        } else {
            apply_rotation(self.p, &self.rotate_local[frame - 1], local, out);
        }
    }

    /// The waves of every box of a level: 6 directions × 2H values per box.
    pub fn outgoing_level(&self, multipoles: &[f64], waves: &mut [f64]) {
        let n = (self.p + 1) * (self.p + 1);
        let h2 = 2 * self.h();
        let mut rotated = vec![0.0; n];
        for (m, w) in multipoles
            .chunks_exact(n)
            .zip(waves.chunks_exact_mut(6 * h2))
        {
            for frame in 0..3 {
                self.multipole_in_frame(frame, m, &mut rotated);
                let (up, rest) = w[2 * frame * h2..(2 * frame + 2) * h2].split_at_mut(h2);
                self.outgoing(&rotated, up, rest);
            }
        }
    }
}

/// The precomputed geometry of a level M2L by plane waves.
pub struct LevelPlan {
    level: Level,
    translation: Translation,
    /// Direct: per target its pairs (source, direction, shift table).
    direct: Vec<Vec<(usize, usize, usize)>>,
    /// Merge-and-shift: per parent its children (box indices, or `usize::MAX` outside
    /// the level... never: uniform), and per direction the entries (source, slot, table)
    /// and the masks of the slots.
    parents: Vec<ParentPlan>,
    /// The shift tables (2H each).
    tables: Vec<Vec<f64>>,
    /// Merge-and-shift: the child shifts, per direction and child (2H each).
    child_tables: Vec<Vec<f64>>,
}

struct ParentPlan {
    children: [usize; 8],
    entries: [Vec<(usize, usize, usize)>; 6],
    masks: [Vec<u8>; 6],
}

impl LevelPlan {
    /// The plan of `level` for `op`.
    pub fn new(op: &PlaneWave, level: Level, translation: Translation) -> Self {
        let mut tables: Vec<Vec<f64>> = Vec::new();
        let mut keys: Vec<[i64; 3]> = Vec::new();
        // Tables keyed by the offset in the frame, in half box widths.
        let mut table = |key: [i64; 3]| -> usize {
            if let Some(i) = keys.iter().position(|&k| k == key) {
                return i;
            }
            keys.push(key);
            tables.push(op.shift_factors(key.map(|x| x as f64 / 2.0)));
            keys.len() - 1
        };
        let mut direct = Vec::new();
        let mut parents = Vec::new();
        let mut child_tables = Vec::new();
        match translation {
            Translation::Direct => {
                let pairs: Vec<Pair> = level.pairs();
                direct = vec![Vec::new(); level.boxes()];
                for pair in pairs {
                    let ct = level.coords(pair.target);
                    let parent = ct.map(|x| x.div_euclid(2));
                    let dir = Direction::for_pair(parent, level.coords(pair.source), ct);
                    let key = dir.rotate(pair.d).map(|x| 2 * x);
                    direct[pair.target].push((pair.source, dir.index(), table(key)));
                }
            }
            Translation::MergeShift => {
                for dir in Direction::ALL {
                    for c in 0..8 {
                        // child centre − parent centre = ±½ per axis.
                        let d = [(c >> 2) & 1, (c >> 1) & 1, c & 1].map(|b| b as f64 - 0.5);
                        child_tables.push(op.signed_shift_factors(dir.rotate(d), dir.is_up()));
                    }
                }
                let half = (level.side() / 2) as i64;
                for pb in 0..(half * half * half) as usize {
                    let parent = crate::geometry::demorton(pb).map(|x| x as i64);
                    let groups = ParentGroups::of(&level, parent);
                    let kids = children(parent).map(|c| level.index(c).unwrap());
                    let mut entries: [Vec<(usize, usize, usize)>; 6] = Default::default();
                    let mut masks: [Vec<u8>; 6] = Default::default();
                    for (dir, list) in groups.sources.iter().enumerate() {
                        masks[dir] = groups.masks(dir);
                        for &(s, mask) in list {
                            let cs = level.coords(s);
                            // parent centre − source centre, half widths: 2(2P + 1) − (2s + 1).
                            let d: [i64; 3] =
                                core::array::from_fn(|i| 4 * parent[i] + 1 - 2 * cs[i]);
                            let key = Direction::ALL[dir].rotate(d);
                            let slot = masks[dir].iter().position(|&m| m == mask).unwrap();
                            entries[dir].push((s, slot, table(key)));
                        }
                    }
                    parents.push(ParentPlan {
                        children: kids,
                        entries,
                        masks,
                    });
                }
            }
        }
        Self {
            level,
            translation,
            direct,
            parents,
            tables,
            child_tables,
        }
    }

    /// The level.
    pub fn level(&self) -> Level {
        self.level
    }

    /// The number of shift tables.
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

/// Scratch of the level M2L: the waves of every box and the per-parent accumulators.
pub struct Scratch {
    /// 6 × 2H per box.
    pub waves: Vec<f64>,
    acc: Vec<f64>,
    incoming: Vec<f64>,
    frame_local: Vec<f64>,
}

impl Scratch {
    /// Scratch for `op` on `level`.
    pub fn new(op: &PlaneWave, level: Level) -> Self {
        let h2 = 2 * op.h();
        let n = (op.p + 1) * (op.p + 1);
        Self {
            waves: vec![0.0; level.boxes() * 6 * h2],
            acc: vec![0.0; 16 * h2],
            incoming: vec![0.0; 16 * h2],
            frame_local: vec![0.0; n],
        }
    }
}

/// Timings of the two parts of a level M2L, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parts {
    /// Multipoles to waves.
    pub outgoing: f64,
    /// Shifts and waves to locals.
    pub incoming: f64,
}

/// Adds the plane-wave M2L of the level's multipoles (scaled, (p + 1)² per box in box
/// order) to the locals.
pub fn level_m2l(
    op: &PlaneWave,
    plan: &LevelPlan,
    scratch: &mut Scratch,
    multipoles: &[f64],
    locals: &mut [f64],
) -> Parts {
    let t0 = std::time::Instant::now();
    op.outgoing_level(multipoles, &mut scratch.waves);
    let t1 = std::time::Instant::now();
    let h2 = 2 * op.h();
    let n = (op.p + 1) * (op.p + 1);
    let Scratch {
        waves,
        acc,
        incoming,
        frame_local,
    } = scratch;
    let wave = |b: usize, dir: usize| &waves[(b * 6 + dir) * h2..(b * 6 + dir + 1) * h2];
    match plan.translation {
        Translation::Direct => {
            for (t, row) in plan.direct.iter().enumerate() {
                let inc = &mut incoming[..6 * h2];
                inc.fill(0.0);
                for &(s, dir, tab) in row {
                    shift_add(
                        &mut inc[dir * h2..(dir + 1) * h2],
                        wave(s, dir),
                        &plan.tables[tab],
                    );
                }
                let local = &mut locals[t * n..(t + 1) * n];
                for frame in 0..3 {
                    frame_local.fill(0.0);
                    let (up, down) = inc[2 * frame * h2..(2 * frame + 2) * h2].split_at(h2);
                    op.incoming(up, down, frame_local);
                    op.local_from_frame(frame, frame_local, local);
                }
            }
        }
        Translation::MergeShift => {
            for parent in &plan.parents {
                for frame in 0..3 {
                    // incoming: 8 children × (up, down).
                    let inc = &mut incoming[..16 * h2];
                    inc.fill(0.0);
                    for half in 0..2 {
                        let dir = 2 * frame + half;
                        let masks = &parent.masks[dir];
                        let acc = &mut acc[..masks.len() * h2];
                        acc.fill(0.0);
                        for &(s, slot, tab) in &parent.entries[dir] {
                            shift_add(
                                &mut acc[slot * h2..(slot + 1) * h2],
                                wave(s, dir),
                                &plan.tables[tab],
                            );
                        }
                        for c in 0..8 {
                            let v = &mut inc[(2 * c + half) * h2..(2 * c + half + 1) * h2];
                            let mut any = false;
                            for (slot, &mask) in masks.iter().enumerate() {
                                if mask & (1 << c) != 0 {
                                    for (x, &a) in
                                        v.iter_mut().zip(&acc[slot * h2..(slot + 1) * h2])
                                    {
                                        *x += a;
                                    }
                                    any = true;
                                }
                            }
                            if any {
                                shift_in_place(v, &plan.child_tables[dir * 8 + c]);
                            }
                        }
                    }
                    for (c, &t) in parent.children.iter().enumerate() {
                        frame_local.fill(0.0);
                        let (up, down) = inc[2 * c * h2..(2 * c + 2) * h2].split_at(h2);
                        op.incoming(up, down, frame_local);
                        op.local_from_frame(frame, frame_local, &mut locals[t * n..(t + 1) * n]);
                    }
                }
            }
        }
    }
    let t2 = std::time::Instant::now();
    Parts {
        outgoing: (t1 - t0).as_secs_f64(),
        incoming: (t2 - t1).as_secs_f64(),
    }
}
