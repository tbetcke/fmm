//! Translations of the Helmholtz family by rotation, coaxial translation and rotation back
//! (RCR; CONVENTIONS §3.15, "Rotations" and "Coaxial translations"; proposal §5.2 and
//! §5.3, Rotation).
//!
//! Each operator translates an expansion about the input frame's centre c to the output
//! frame's centre c′ along t = c′ − c, in three steps:
//!
//! 1. Rotate the input by the rotation Q with Q t = |t| ẑ: with θ and φ the polar angles
//!    of t, Qᵀ = R_z(φ) R_y(θ), and coefficients rotate by Oⁿ(Q) = Oⁿ(R_y(θ))ᵀ Oⁿ(R_z(−φ))
//!    (§3.15, "Rotations": Yₙ(Q x̂) = Oⁿ(Q) Yₙ(x̂), Oⁿ orthogonal): a 2 × 2 rotation per
//!    order, then the transposed y-block of θ from
//!    [`y_blocks`], whose cos half acts on
//!    the slots m ≥ 0 and sin half on the slots m < 0.
//! 2. Apply the coaxial table per order, the same coefficient to the slots +m and −m,
//!    from the input's degrees n ≤ p_in to the output's n′ ≤ p_out (rectangular).
//! 3. Rotate back by Oⁿ(Qᵀ) = Oⁿ(R_z(φ)) Oⁿ(R_y(θ)): the y-block, then the 2 × 2
//!    rotations.
//!
//! On the axis (t along ±ẑ) θ is 0 or π and φ = 0; the blocks of θ = π are those of the
//! recursion at cos θ = −1, so no case is special.
//!
//! # Stored form
//!
//! With the tables of [`CoaxialTable`] in §3.15's stored normalisation (Ê = e^{−Im z} E for
//! (S|S) = (R|R), Ê = e^{Im z} s^{n+n′} E for (S|R), z = k|t|) and the stored coefficients
//! M̂ = e^{−β} M / sⁿ, L̂ = e^β L sⁿ of the [`leaf`](super::leaf) operators, the coaxial
//! step of each operator is
//!
//! ```text
//! M2M  child c → parent P, |t| = a_c:  M̂ᴾ_{n′} += Σₙ (s_cⁿ / s_P^{n′}) Ê_{n′n} M̂ᶜₙ
//! L2L  parent P → child c, |t| = a_c:  L̂ᶜ_{n′} += Σₙ (s_c^{n′} / s_Pⁿ) Ê_{n′n} L̂ᴾₙ
//! M2L  one level:                      L̂_{n′}   += 4π ik e^{2β − Im(k)|t|} Σₙ Ê_{n′n} M̂ₙ
//! ```
//!
//! the damping factors of M2M and L2L cancelling (β_P = 2β_c, Im z = β_c). The Yukawa
//! forms act on M̂^Y = i⁻ⁿ M̂ and L̂^Y = iⁿ L̂ through the real tables i^{n−n′} Ê (M2M),
//! i^{n′−n} Ê (L2L) and −i^{n+n′} Ê (M2L, with 4πκ in place of 4π ik); the operators read
//! the real part of the product.
//!
//! The tables come from [`m2m_table`], [`l2l_table`] and [`m2l_table`], which build them by
//! §3.15's rule ([`CoaxialTable::m2m`], [`CoaxialTable::l2l`], [`CoaxialTable::m2l`]) from
//! the levels and frames. A table carries no record of its shift: one built for another
//! shift gives a wrong result without a panic. The operators check its degrees.
//!
//! # Storage
//!
//! As the [`leaf`](super::leaf) operators: a complex expansion is two real planes of
//! (p + 1)² (real, then imaginary) in the layout idx(n, m) of §3.6; Yukawa's real path has
//! one plane of (p + 1)². Every operator adds (+=) into its output and none applies
//! 1/(4π). The y-blocks are built per call into the [`Workspace`], O(p³).
//!
//! # Tested range
//!
//! `tests/helmholtz/translation_*.rs` checks f64 (Helmholtz CH1.3c): against [`direct`](super::direct)
//! for real, complex, weakly damped and imaginary k at |k| a from 0.5 to 30, above the
//! projection threshold at Re(k) a = 50 (real and weakly damped k) and at Im(k) a = 100
//! (Re(k) a = 20, and imaginary k), over the eight children, all 316 offsets of the V list
//! (|k| a = 5) and rectangular degrees, in the FMM-weighted measure of the tests; the
//! chains P2M → M2M → M2P, P2L → L2L → L2P and P2M → M2L → L2P against the direct sum
//! to p = 80; the limit k → 0 against Laplace's [`rotation`](crate::rotation); Yukawa
//! two ways to κ a = 300. Under strong damping an M2L's output is smaller than its input
//! by about e^{−Im(k)(|t| − 2a)} and the coaxial sum cancels: the output's error relative
//! to itself grows (about 1e-10 where that factor is 1e-19) while it stays at round-off
//! relative to the input, and the pruning of proposal §5.5 removes such contributions; an
//! L2L to a child on the far side of the parent from the sources shows the same at
//! Im(k) a = 100 (about 1e-11). f32 is checked at p = 8 only (CONVENTIONS §3.15 reserves
//! it).
//!
//! M2M and L2L apply the power scales s_cⁿ / s_P^{n′} as one factor per entry, so they are
//! limited at low frequency to degrees with p log₁₀(1/s_P) below about 300 (for example
//! p ≤ 50 at s_P = 10⁻⁶); beyond that the factor overflows and the output is not finite,
//! and the unscaled (S|S) entries, of order (k a_c)^{|n′ − n|}, underflow. An FMM's low
//! degrees there are far smaller.
//!
//! # Panics
//!
//! Every operator panics if the workspace was built for a smaller degree than
//! max(p_in, p_out), if a frame's radius is not its level's half-width, if the levels have
//! different k, if a coefficient slice has the wrong length, and if the table's degrees
//! are not (p_in, p_out). M2M and L2L panic unless the frames are a parent and one of its
//! children (r_P = 2 r_c and every |c′ᵢ − cᵢ| = r_c); M2L panics if the shift is zero. The
//! Yukawa forms panic if the level is not Yukawa.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::Frame;
//! use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, p2p, rcr};
//!
//! let p = 24;
//! let k = Complex::new(3.0, 0.2);
//! let level = Level::new(k, 0.5);
//! let source = Frame::new([0.0; 3], 0.5);
//! let target = Frame::new([2.0, 1.0, -1.0], 0.5);
//! let mut ws = Workspace::new(p);
//!
//! let sources = [[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]];
//! let charges = [Complex::new(1.0, 0.5), Complex::new(-0.5, 0.0)];
//! let mut multipole = vec![0.0; 2 * (p + 1) * (p + 1)];
//! leaf::p2m(p, &level, &source, &sources, &charges, &mut ws, &mut multipole);
//!
//! let table = rcr::m2l_table(p, p, &level, &source, &target);
//! let mut local = vec![0.0; 2 * (p + 1) * (p + 1)];
//! rcr::m2l(p, p, &level, &source, &target, &table, &mut ws, &multipole, &mut local);
//!
//! let x = [[2.2, 0.9, -0.8]];
//! let mut potential = [Complex::new(0.0, 0.0)];
//! leaf::l2p(p, &level, &target, &local, &x, &mut ws, &mut potential, None);
//! let mut exact = [Complex::new(0.0, 0.0)];
//! p2p::direct_sum(k, &sources, &charges, &x, &mut exact, None);
//! assert!((potential[0] - exact[0]).norm() < 1e-8 * exact[0].norm());
//! ```

use nd_fmm_math::coaxial::CoaxialTable;
use nd_fmm_math::orthonormal::rotation::{compact_offset, y_blocks};
use nd_fmm_math::{Complex, Layout, RealScalar};

use super::leaf::{check_coefficients, check_frame, check_yukawa, four_pi, four_pi_ik};
use super::{Level, Workspace};
use crate::Frame;

/// The M2M table (S|S) from a child of `child`'s level (degree `p_child`) to its parent
/// (degree `p_parent`), at the shift a_c, by §3.15's rule ([`CoaxialTable::m2m`]).
///
/// # Panics
///
/// As [`CoaxialTable::m2m`].
pub fn m2m_table<T: RealScalar>(
    p_child: usize,
    p_parent: usize,
    child: &Level<T>,
) -> CoaxialTable<T> {
    CoaxialTable::m2m(child.k(), child.circumradius(), p_child, p_parent)
}

/// The L2L table (R|R) from a parent (degree `p_parent`) to a child of `child`'s level
/// (degree `p_child`), at the shift a_c, by §3.15's rule ([`CoaxialTable::l2l`]).
///
/// # Panics
///
/// As [`CoaxialTable::l2l`].
pub fn l2l_table<T: RealScalar>(
    p_parent: usize,
    p_child: usize,
    child: &Level<T>,
) -> CoaxialTable<T> {
    CoaxialTable::l2l(child.k(), child.circumradius(), p_parent, p_child)
}

/// The M2L table (S|R) between the boxes `from` and `to` of `level`, at the distance of
/// their centres, by §3.15's rule ([`CoaxialTable::m2l`]).
///
/// # Panics
///
/// As [`CoaxialTable::m2l`], which needs the distance to exceed the circumradius.
pub fn m2l_table<T: RealScalar>(
    p_src: usize,
    p_tgt: usize,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
) -> CoaxialTable<T> {
    let t = norm(shift(from, to));
    CoaxialTable::m2l(
        level.k(),
        t,
        level.circumradius(),
        level.scale(),
        p_src,
        p_tgt,
    )
}

/// M2M: adds the stored complex multipole `multipole_in` of a child box, degree
/// `p_child`, translated to its parent, degree `p_parent`, to `multipole_out` (module
/// documentation). `table` is [`m2m_table`]`(p_child, p_parent, child)`.
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn m2m<T: RealScalar>(
    p_child: usize,
    p_parent: usize,
    child: &Level<T>,
    parent: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    multipole_in: &[T],
    multipole_out: &mut [T],
) {
    check_pair(child, parent, from, to);
    check_table(table, p_child, p_parent);
    check_coefficients(p_child, multipole_in, 2, "multipole_in");
    check_coefficients(p_parent, multipole_out, 2, "multipole_out");
    let (sc, sp) = (child.scale(), parent.scale());
    translate(
        p_child,
        p_parent,
        shift(from, to),
        ws,
        multipole_in,
        multipole_out,
        2,
        |m, n_out, n_in| table.get(m, n_out, n_in) * power_ratio(sc, n_in, sp, n_out),
    );
}

/// L2L: adds the stored complex local `local_in` of a parent box, degree `p_parent`,
/// translated to one of its children, degree `p_child`, to `local_out` (module
/// documentation). `table` is [`l2l_table`]`(p_parent, p_child, child)`.
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn l2l<T: RealScalar>(
    p_parent: usize,
    p_child: usize,
    parent: &Level<T>,
    child: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    local_in: &[T],
    local_out: &mut [T],
) {
    check_pair(child, parent, to, from);
    check_table(table, p_parent, p_child);
    check_coefficients(p_parent, local_in, 2, "local_in");
    check_coefficients(p_child, local_out, 2, "local_out");
    let (sc, sp) = (child.scale(), parent.scale());
    translate(
        p_parent,
        p_child,
        shift(from, to),
        ws,
        local_in,
        local_out,
        2,
        |m, n_out, n_in| table.get(m, n_out, n_in) * power_ratio(sc, n_out, sp, n_in),
    );
}

/// M2L: adds the stored complex multipole `multipole` of the box `from`, degree `p_src`,
/// translated to a local about the box `to` of the same `level`, degree `p_tgt`, to
/// `local` (module documentation). `table` is [`m2l_table`]`(p_src, p_tgt, level, from,
/// to)`. The convergence of the result (boxes well separated) is the caller's to meet.
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn m2l<T: RealScalar>(
    p_src: usize,
    p_tgt: usize,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    let t = check_level_pair(level, from, to);
    check_table(table, p_src, p_tgt);
    check_coefficients(p_src, multipole, 2, "multipole");
    check_coefficients(p_tgt, local, 2, "local");
    let k = level.k();
    let factor = four_pi_ik(k) * (level.damping() + level.damping() - k.im * norm(t)).exp();
    translate(
        p_src,
        p_tgt,
        t,
        ws,
        multipole,
        local,
        2,
        |m, n_out, n_in| table.get(m, n_out, n_in) * factor,
    );
}

/// Yukawa M2M: as [`m2m`] on the real stored multipole M̂^Y, with the real table
/// i^{n−n′} Ê (module documentation). The levels must be Yukawa.
///
/// # Panics
///
/// As [`m2m`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn m2m_yukawa<T: RealScalar>(
    p_child: usize,
    p_parent: usize,
    child: &Level<T>,
    parent: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    multipole_in: &[T],
    multipole_out: &mut [T],
) {
    check_yukawa(child);
    check_pair(child, parent, from, to);
    check_table(table, p_child, p_parent);
    check_coefficients(p_child, multipole_in, 1, "multipole_in");
    check_coefficients(p_parent, multipole_out, 1, "multipole_out");
    let (sc, sp) = (child.scale(), parent.scale());
    translate(
        p_child,
        p_parent,
        shift(from, to),
        ws,
        multipole_in,
        multipole_out,
        1,
        |m, n_out, n_in| {
            let e =
                real_part_times_i_power(n_in as isize - n_out as isize, table.get(m, n_out, n_in));
            Complex::new(e * power_ratio(sc, n_in, sp, n_out), T::zero())
        },
    );
}

/// Yukawa L2L: as [`l2l`] on the real stored local L̂^Y, with the real table i^{n′−n} Ê
/// (module documentation). The levels must be Yukawa.
///
/// # Panics
///
/// As [`l2l`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn l2l_yukawa<T: RealScalar>(
    p_parent: usize,
    p_child: usize,
    parent: &Level<T>,
    child: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    local_in: &[T],
    local_out: &mut [T],
) {
    check_yukawa(child);
    check_pair(child, parent, to, from);
    check_table(table, p_parent, p_child);
    check_coefficients(p_parent, local_in, 1, "local_in");
    check_coefficients(p_child, local_out, 1, "local_out");
    let (sc, sp) = (child.scale(), parent.scale());
    translate(
        p_parent,
        p_child,
        shift(from, to),
        ws,
        local_in,
        local_out,
        1,
        |m, n_out, n_in| {
            let e =
                real_part_times_i_power(n_out as isize - n_in as isize, table.get(m, n_out, n_in));
            Complex::new(e * power_ratio(sc, n_out, sp, n_in), T::zero())
        },
    );
}

/// Yukawa M2L: as [`m2l`] on the real stored expansions, with the real table −i^{n+n′} Ê
/// and 4πκ e^{2β − κ|t|} (module documentation). The level must be Yukawa.
///
/// # Panics
///
/// As [`m2l`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn m2l_yukawa<T: RealScalar>(
    p_src: usize,
    p_tgt: usize,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    table: &CoaxialTable<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    let kappa = check_yukawa(level);
    let t = check_level_pair(level, from, to);
    check_table(table, p_src, p_tgt);
    check_coefficients(p_src, multipole, 1, "multipole");
    check_coefficients(p_tgt, local, 1, "local");
    let factor =
        four_pi::<T>() * kappa * (level.damping() + level.damping() - kappa * norm(t)).exp();
    translate(
        p_src,
        p_tgt,
        t,
        ws,
        multipole,
        local,
        1,
        |m, n_out, n_in| {
            let e = real_part_times_i_power((n_in + n_out) as isize, table.get(m, n_out, n_in));
            Complex::new(-(e * factor), T::zero())
        },
    );
}

/// The three steps of the module documentation, for `planes` = 2 (a complex expansion)
/// or 1 (a real one, for which `entry` must be real). `entry(m, n′, n)` is the coaxial
/// step's full coefficient from degree n to n′ of order m.
#[allow(clippy::too_many_arguments)]
fn translate<T: RealScalar>(
    p_in: usize,
    p_out: usize,
    t: [T; 3],
    ws: &mut Workspace<T>,
    input: &[T],
    output: &mut [T],
    planes: usize,
    entry: impl Fn(usize, usize, usize) -> Complex<T>,
) {
    let p = p_in.max(p_out);
    let buffers = ws.translation(p);
    let angles = Angles::of(t);
    y_blocks(
        p,
        angles.cos_theta,
        angles.sin_theta,
        buffers.blocks,
        buffers.rotation,
    );
    let (len_in, len_out) = (Layout::new(p_in).len(), Layout::new(p_out).len());

    // 1. Rotate each plane of the input: z by −φ, then the transposed y-block.
    let (rotated, work) = (&mut *buffers.second, &mut *buffers.first);
    for plane in 0..planes {
        let x = &mut work[..len_in];
        x.copy_from_slice(&input[plane * len_in..(plane + 1) * len_in]);
        rotate_z(p_in, &angles, -T::one(), x);
        apply_y(
            p_in,
            buffers.blocks,
            true,
            x,
            &mut rotated[plane * len_in..(plane + 1) * len_in],
        );
    }

    // 2. The coaxial step, per order, into `work`.
    let shifted = &mut work[..planes * len_out];
    shifted.fill(T::zero());
    for n_out in 0..=p_out {
        for m in 0..=n_out.min(p_in) {
            for slot in slots(m) {
                let i_out = index(n_out, slot);
                let mut sum = Complex::new(T::zero(), T::zero());
                for n_in in m..=p_in {
                    let i_in = index(n_in, slot);
                    let c = if planes == 2 {
                        Complex::new(rotated[i_in], rotated[len_in + i_in])
                    } else {
                        Complex::new(rotated[i_in], T::zero())
                    };
                    let e = entry(m, n_out, n_in);
                    sum = sum
                        + if planes == 2 {
                            e * c
                        } else {
                            Complex::new(e.re * c.re, T::zero())
                        };
                }
                shifted[i_out] = sum.re;
                if planes == 2 {
                    shifted[len_out + i_out] = sum.im;
                }
            }
        }
    }

    // 3. Rotate back: the y-block, then z by φ, and accumulate.
    for plane in 0..planes {
        let y = &mut rotated[..len_out];
        apply_y(
            p_out,
            buffers.blocks,
            false,
            &shifted[plane * len_out..(plane + 1) * len_out],
            y,
        );
        rotate_z(p_out, &angles, T::one(), y);
        for (o, &v) in output[plane * len_out..(plane + 1) * len_out]
            .iter_mut()
            .zip(y.iter())
        {
            *o = *o + v;
        }
    }
}

/// The slots +m and −m of order m (one slot for m = 0).
fn slots(m: usize) -> impl Iterator<Item = isize> {
    let m = m as isize;
    core::iter::once(m).chain((m > 0).then_some(-m))
}

/// idx(n, m) of §3.6.
fn index(n: usize, m: isize) -> usize {
    (n * n + n).wrapping_add_signed(m)
}

/// The polar angles of a shift: cos θ, sin θ ≥ 0, cos φ, sin φ (φ = 0 on the axis).
struct Angles<T> {
    cos_theta: T,
    sin_theta: T,
    cos_phi: T,
    sin_phi: T,
}

impl<T: RealScalar> Angles<T> {
    fn of([x, y, z]: [T; 3]) -> Self {
        let rho = x.hypot(y);
        let r = rho.hypot(z);
        let (cos_phi, sin_phi) = if rho > T::zero() {
            (x / rho, y / rho)
        } else {
            (T::one(), T::zero())
        };
        Self {
            cos_theta: z / r,
            sin_theta: rho / r,
            cos_phi,
            sin_phi,
        }
    }
}

/// Applies Oⁿ(R_z(σφ)), n ≤ p, in place: for m > 0, with c = cos mσφ and s = sin mσφ,
/// slot +m ← c x₊ − s x₋ and slot −m ← s x₊ + c x₋ (as
/// [`blocks`](nd_fmm_math::orthonormal::rotation::blocks)).
fn rotate_z<T: RealScalar>(p: usize, angles: &Angles<T>, sigma: T, x: &mut [T]) {
    let step = (angles.cos_phi, sigma * angles.sin_phi);
    let mut phase = (T::one(), T::zero());
    for m in 1..=p {
        phase = (
            phase.0 * step.0 - phase.1 * step.1,
            phase.1 * step.0 + phase.0 * step.1,
        );
        let (c, s) = phase;
        for n in m..=p {
            let (plus, minus) = (index(n, m as isize), index(n, -(m as isize)));
            let (a, b) = (x[plus], x[minus]);
            x[plus] = c * a - s * b;
            x[minus] = s * a + c * b;
        }
    }
}

/// `out` = Oⁿ(R_y(θ)) x per degree n ≤ p, or its transpose with `transpose`, from the
/// compact blocks (cos half over the slots m ≥ 0, sin half over m < 0).
fn apply_y<T: RealScalar>(p: usize, blocks: &[T], transpose: bool, x: &[T], out: &mut [T]) {
    for n in 0..=p {
        let block = &blocks[compact_offset(n)..compact_offset(n + 1)];
        let (cos, sin) = block.split_at((n + 1) * (n + 1));
        for a in 0..=n {
            let mut sum = T::zero();
            for b in 0..=n {
                let e = if transpose {
                    cos[b * (n + 1) + a]
                } else {
                    cos[a * (n + 1) + b]
                };
                sum = sum + e * x[index(n, b as isize)];
            }
            out[index(n, a as isize)] = sum;
        }
        for a in 1..=n {
            let mut sum = T::zero();
            for b in 1..=n {
                let e = if transpose {
                    sin[(b - 1) * n + (a - 1)]
                } else {
                    sin[(a - 1) * n + (b - 1)]
                };
                sum = sum + e * x[index(n, -(b as isize))];
            }
            out[index(n, -(a as isize))] = sum;
        }
    }
}

/// Re(iʲ e).
fn real_part_times_i_power<T: RealScalar>(j: isize, e: Complex<T>) -> T {
    match j.rem_euclid(4) {
        0 => e.re,
        1 => -e.im,
        2 => -e.re,
        _ => e.im,
    }
}

/// aⁿ / b^{n′}, formed as (a/b)ⁿ b^{n − n′}; b^{n − n′} overflows where |n − n′| log₁₀(1/b)
/// exceeds about 300 (module documentation, "Tested range").
fn power_ratio<T: RealScalar>(a: T, n: usize, b: T, n_prime: usize) -> T {
    (a / b).powi(n as i32) * b.powi(n as i32 - n_prime as i32)
}

/// c′ − c.
pub(super) fn shift<T: RealScalar>(from: &Frame<T>, to: &Frame<T>) -> [T; 3] {
    core::array::from_fn(|i| to.centre[i] - from.centre[i])
}

pub(super) fn norm<T: RealScalar>([x, y, z]: [T; 3]) -> T {
    x.hypot(y).hypot(z)
}

/// The frames of a parent and one of its children, on their levels.
fn check_pair<T: RealScalar>(
    child: &Level<T>,
    parent: &Level<T>,
    child_frame: &Frame<T>,
    parent_frame: &Frame<T>,
) {
    check_frame(child, child_frame);
    check_frame(parent, parent_frame);
    check_same_k(child, parent);
    let r = child.half_width();
    assert!(
        parent.half_width() == r + r,
        "the parent's half-width must be twice the child's"
    );
    let tolerance = T::epsilon().sqrt() * r;
    let t = shift(child_frame, parent_frame);
    assert!(
        t.iter().all(|&d| (d.abs() - r).abs() <= tolerance),
        "the frames are not a parent and one of its children"
    );
}

/// Both frames on `level`, apart; returns the shift.
fn check_level_pair<T: RealScalar>(level: &Level<T>, from: &Frame<T>, to: &Frame<T>) -> [T; 3] {
    check_frame(level, from);
    check_frame(level, to);
    let t = shift(from, to);
    assert!(norm(t) > T::zero(), "the M2L needs distinct centres");
    t
}

pub(super) fn check_same_k<T: RealScalar>(a: &Level<T>, b: &Level<T>) {
    assert!(a.k() == b.k(), "the two levels have different wavenumbers");
}

fn check_table<T: RealScalar>(table: &CoaxialTable<T>, p_in: usize, p_out: usize) {
    assert!(
        table.p_src() == p_in && table.p_tgt() == p_out,
        "the table has degrees {} → {}, expected {p_in} → {p_out}",
        table.p_src(),
        table.p_tgt()
    );
}
