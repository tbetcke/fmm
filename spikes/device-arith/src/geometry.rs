//! Leaf-scaled geometry of CONVENTIONS §3.13 on the host, and the pairs of
//! `tools/fixtures/check_p2p_domain.py`, rebuilt in Rust from its description (no
//! fixture files).
//!
//! Keys are (level, index) on levels 0–16; the octree's `points_to_morton` is emulated as
//! the Python script emulates it (`leaf_index`), the stored coordinates are evaluated in
//! f64 in the order of §3.13 ("Leaf-scaled coordinates") and then rounded to T, and the
//! relative frames come from integer centres ("Relative frames").

use nd_fmm_validate::SplitMix64;

/// The deepest level of the octree.
pub const DEEPEST_LEVEL: u32 = 16;

/// Ulps on either side of every adversarial point (the script's `WINDOW`).
const WINDOW: i64 = 4;

/// A box: its level and its index along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    /// The level, 0–16.
    pub level: u32,
    /// The index along x, y and z, each below 2^level.
    pub index: [u32; 3],
}

/// The integer centre of §3.13 along one axis: (2i + 1) 2^(L − l).
fn integer_centre(index: u32, level: u32, reference: u32) -> i64 {
    (2 * i64::from(index) + 1) << (reference - level)
}

/// The relative frame (ĉ(s|t), r̂(s|t)) of §3.13, exact in f64 (and in f32).
pub fn relative_frame(s: Key, t: Key) -> ([f64; 3], f64) {
    let reference = s.level.max(t.level);
    let scale = 2f64.powi(t.level as i32 - reference as i32);
    let c = std::array::from_fn(|k| {
        let n = integer_centre(s.index[k], s.level, reference)
            - integer_centre(t.index[k], t.level, reference);
        n as f64 * scale
    });
    (c, 2f64.powi(t.level as i32 - s.level as i32))
}

/// A cubic domain as `nd_fmm_exec::geometry::Domain` sees it: the box `lo`, `hi` that
/// `points_to_morton` uses (one side per axis), and the side w = max_k fl(hi_k − lo_k).
#[derive(Clone, Debug)]
pub struct Domain {
    /// The name in the report.
    pub name: &'static str,
    /// The lower corner.
    pub lo: [f64; 3],
    /// The upper corner.
    pub hi: [f64; 3],
    /// The side w.
    pub w: f64,
}

impl Domain {
    fn new(name: &'static str, lo: [f64; 3], hi: [f64; 3]) -> Self {
        let w = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        Self { name, lo, hi, w }
    }

    /// The index along axis k of the leaf on `level` that `points_to_morton` gives a point
    /// with coordinate x (`morton::from_physical_point` at level 16, then the ancestor).
    pub fn leaf_index(&self, x: f64, k: usize, level: u32) -> u32 {
        let reference = (x - self.lo[k]) / (self.hi[k] - self.lo[k]);
        let deepest =
            ((reference * 2f64.powi(DEEPEST_LEVEL as i32)) as u64).min((1 << DEEPEST_LEVEL) - 1);
        (deepest >> (DEEPEST_LEVEL - level)) as u32
    }

    /// The stored u of §3.13 along axis k, in f64: fl(fl(fl(x − a) 2^(l+1)) / w) − (2i + 1).
    pub fn stored_u(&self, x: f64, k: usize, level: u32, index: u32) -> f64 {
        (x - self.lo[k]) * 2f64.powi(level as i32 + 1) / self.w - (2.0 * f64::from(index) + 1.0)
    }

    /// The double nearest to c + p r along axis k, with c the centre and r the half-width of
    /// the leaf `key`: a + ((2i + 1 + p) / 2^(l+1)) w, rounded once by an fma (p dyadic).
    fn position(&self, k: usize, key: Key, p: f64) -> f64 {
        let scaled = (2.0 * f64::from(key.index[k]) + 1.0 + p) / 2f64.powi(key.level as i32 + 1);
        scaled.mul_add(self.w, self.lo[k])
    }

    /// The spacing of fl(x − a) near x: the ulp of the largest of |x|, |a_k|, |hi_k|.
    fn coarse_ulp(&self, x: f64, k: usize) -> f64 {
        ulp(x.abs().max(self.lo[k].abs()).max(self.hi[k].abs()))
    }

    /// The smallest double near x0 that `points_to_morton` puts above index `below` on
    /// `level` along axis k: the leaf face as the octree draws it (the script's `face`).
    fn face(&self, x0: f64, k: usize, level: u32, below: u32) -> Option<f64> {
        let span = 64.0 * self.coarse_ulp(x0, k);
        let lo = (x0 - span).max(next_up(self.lo[k]));
        let hi = (x0 + span).min(next_down(self.hi[k]));
        if !(self.leaf_index(lo, k, level) <= below && below < self.leaf_index(hi, k, level)) {
            return None;
        }
        let (mut a, mut b) = (ordered(lo), ordered(hi));
        while b - a > 1 {
            let mid = a + (b - a) / 2;
            if self.leaf_index(from_ordered(mid), k, level) > below {
                b = mid;
            } else {
                a = mid;
            }
        }
        Some(from_ordered(b))
    }

    /// x, its `WINDOW` neighbours on either side, and `WINDOW` steps of the spacing of
    /// fl(x − a) on either side.
    fn window(&self, x: f64, k: usize) -> Vec<f64> {
        let mut xs = vec![x];
        let (mut lo, mut hi) = (x, x);
        for _ in 0..WINDOW {
            lo = next_down(lo);
            hi = next_up(hi);
            xs.extend([lo, hi]);
        }
        let step = self.coarse_ulp(x, k);
        xs.extend((-WINDOW..=WINDOW).map(|j| x + j as f64 * step));
        xs
    }

    /// The stored u (in f64) of the doubles near the scaled positions p of the leaf `key`
    /// along axis k (x = c + p r, and at p = ±1 also the face as `points_to_morton` draws
    /// it) that `points_to_morton` puts into the leaf (the script's `leaf_values`).
    pub fn leaf_values(&self, k: usize, key: Key, positions: &[f64]) -> Vec<f64> {
        let index = key.index[k];
        let mut centres = Vec::new();
        for &p in positions {
            let x = self.position(k, key, p);
            centres.push(x);
            if p.abs() == 1.0 {
                let below = if p == 1.0 {
                    Some(index)
                } else {
                    index.checked_sub(1)
                };
                if let Some(face) = below.and_then(|b| self.face(x, k, key.level, b)) {
                    centres.push(face);
                }
            }
        }
        let mut xs: Vec<f64> = centres.iter().flat_map(|&c| self.window(c, k)).collect();
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        xs.into_iter()
            .filter(|&x| self.lo[k] < x && x < self.hi[k])
            .filter(|&x| self.leaf_index(x, k, key.level) == index)
            .map(|x| self.stored_u(x, k, key.level, index))
            .collect()
    }

    /// A double point uniform in the leaf `key` that `points_to_morton` puts into it.
    pub fn random_point(&self, rng: &mut SplitMix64, key: Key) -> [f64; 3] {
        loop {
            let x: [f64; 3] = std::array::from_fn(|k| self.position(k, key, rng.range(-1.0, 1.0)));
            if (0..3).all(|k| {
                self.lo[k] < x[k]
                    && x[k] < self.hi[k]
                    && self.leaf_index(x[k], k, key.level) == key.index[k]
            }) {
                return x;
            }
        }
    }

    /// The stored u of the point x in the leaf `key`, in f64.
    pub fn stored(&self, x: [f64; 3], key: Key) -> [f64; 3] {
        std::array::from_fn(|k| self.stored_u(x[k], k, key.level, key.index[k]))
    }
}

/// The ulp of a non-negative finite x, as Python's `math.ulp`.
fn ulp(x: f64) -> f64 {
    next_up(x) - x
}

fn next_up(x: f64) -> f64 {
    from_ordered(ordered(x) + 1)
}

fn next_down(x: f64) -> f64 {
    from_ordered(ordered(x) - 1)
}

/// The position of the double x in the order of the doubles.
fn ordered(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits >= 0 {
        bits
    } else {
        -(bits & 0x7FFF_FFFF_FFFF_FFFF)
    }
}

fn from_ordered(n: i64) -> f64 {
    if n >= 0 {
        f64::from_bits(n as u64)
    } else {
        f64::from_bits((-n) as u64 | 1 << 63)
    }
}

/// The bounding box of points, as `nd_octree::octree::compute_global_bounding_box` forms it
/// on one rank.
fn bounding_box(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let lo: [f64; 3] =
        std::array::from_fn(|k| points.iter().map(|x| x[k]).fold(f64::INFINITY, f64::min));
    let hi: [f64; 3] = std::array::from_fn(|k| {
        points
            .iter()
            .map(|x| x[k])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let diam: [f64; 3] = std::array::from_fn(|k| hi[k] - lo[k]);
    let mean: [f64; 3] = std::array::from_fn(|k| lo[k] + 0.5 * diam[k]);
    let max_diam = diam.iter().copied().fold(0.0, f64::max) * (1.0 + 1.0 / 65536.0);
    (
        std::array::from_fn(|k| mean[k] - 0.5 * max_diam),
        std::array::from_fn(|k| mean[k] + 0.5 * max_diam),
    )
}

fn cloud(rng: &mut SplitMix64, centre: [f64; 3], spread: f64, count: usize) -> Vec<[f64; 3]> {
    (0..count)
        .map(|_| std::array::from_fn(|k| centre[k] + spread * rng.range(-1.0, 1.0)))
        .collect()
}

/// The four domains of the script: dyadic at the origin, dyadic, generic, and far from
/// the origin (the last two from the bounding boxes of seeded clouds).
pub fn domains(rng: &mut SplitMix64) -> Vec<Domain> {
    let (glo, ghi) = bounding_box(&cloud(rng, [0.2, -0.35, 0.05], 1.3, 200));
    let far_centre = [3.7e5 + 0.3, -1.2e6 - 0.7, 8.1e5 + 0.1];
    let (flo, fhi) = bounding_box(&cloud(rng, far_centre, 37.25, 200));
    let a = [-1.0, 0.5, 0.75];
    vec![
        Domain::new("dyadic at the origin", [0.0; 3], [1.0; 3]),
        Domain::new("dyadic", a, a.map(|c| c + 4.0)),
        Domain::new("generic", glo, ghi),
        Domain::new("far from the origin", flo, fhi),
    ]
}

/// `interaction_manager::is_adjacent`: the closed cubes touch and neither contains the other.
fn adjacent(a: Key, b: Key) -> bool {
    let level = a.level.max(b.level);
    let (sa, sb) = (level - a.level, level - b.level);
    if a.level <= b.level && (0..3).all(|k| b.index[k] >> (b.level - a.level) == a.index[k]) {
        return false;
    }
    if b.level <= a.level && (0..3).all(|k| a.index[k] >> (a.level - b.level) == b.index[k]) {
        return false;
    }
    (0..3).all(|k| {
        let (a_min, b_min) = (i64::from(a.index[k]) << sa, i64::from(b.index[k]) << sb);
        let (a_max, b_max) = (a_min + (1 << sa) - 1, b_min + (1 << sb) - 1);
        a_min <= b_max + 1 && b_min <= a_max + 1
    })
}

/// Every key on levels l_t − 1, l_t, l_t + 1 adjacent to t: the U-list entries a
/// 2:1-balanced tree can give t.
pub fn u_list(t: Key) -> Vec<Key> {
    let mut out = Vec::new();
    for ls in t.level.saturating_sub(1)..=(t.level + 1).min(DEEPEST_LEVEL) {
        let n = 1i64 << ls;
        let ranges: Vec<Vec<u32>> = (0..3)
            .map(|k| {
                let it = i64::from(t.index[k]);
                let centre = if ls < t.level {
                    it >> 1
                } else if ls > t.level {
                    2 * it
                } else {
                    it
                };
                (centre - 2..centre + 4)
                    .filter(|j| (0..n).contains(j))
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        for &x in &ranges[0] {
            for &y in &ranges[1] {
                for &z in &ranges[2] {
                    let s = Key {
                        level: ls,
                        index: [x, y, z],
                    };
                    if adjacent(s, t) {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

/// The target leaves of a level: first, last, middle and a random index on the diagonal,
/// and one key mixing them.
pub fn targets(rng: &mut SplitMix64, level: u32) -> Vec<Key> {
    let n = 1u32 << level;
    let mut patterns = vec![
        0,
        1 % n,
        n - 1,
        n.saturating_sub(2),
        n / 2,
        (n / 2).saturating_sub(1),
        (rng.next_u64() % u64::from(n)) as u32,
    ];
    patterns.sort();
    patterns.dedup();
    let mut keys: Vec<Key> = patterns
        .iter()
        .map(|&i| Key {
            level,
            index: [i; 3],
        })
        .collect();
    let mixed =
        std::array::from_fn(|_| patterns[(rng.next_u64() % patterns.len() as u64) as usize]);
    keys.push(Key {
        level,
        index: mixed,
    });
    keys
}

/// Scaled positions, in the frame of t, where the closed leaves s and t meet along one
/// axis: both ends, the midpoint, and t's centre (the cancellation ĉ + r̂ u ≈ 0), s's
/// centre and ±½ where they lie inside (the script's `locus`).
pub fn locus(c_hat: f64, r_hat: f64) -> Vec<f64> {
    let (lo, hi) = ((c_hat - r_hat).max(-1.0), (c_hat + r_hat).min(1.0));
    let mut points = vec![lo, hi, (lo + hi) / 2.0];
    for p in [0.0, c_hat, 0.5, -0.5] {
        if lo <= p && p <= hi {
            points.push(p);
        }
    }
    points.sort_by(f64::total_cmp);
    points.dedup();
    points
}

/// The positions of a self pair (s = t) along every axis.
pub const SELF_POSITIONS: [f64; 6] = [0.0, 1.0, -1.0, 0.5, -0.5, 1.0 / 3.0];

#[cfg(test)]
mod tests {
    use super::*;

    /// The frames of the 2:1 U list are those of §3.13, "Fast kernels": ĉ ∈ {±1, ±3} with
    /// r̂ = 2, {0, ±2} with r̂ = 1, {±½, ±3/2} with r̂ = ½. Error measure: exact equality.
    #[test]
    fn u_list_frames() {
        let t = Key {
            level: 5,
            index: [7, 8, 9],
        };
        let list = u_list(t);
        assert!(!list.is_empty());
        for s in list {
            let (c, r) = relative_frame(s, t);
            let allowed: &[f64] = match s.level as i32 - t.level as i32 {
                -1 => &[-3.0, -1.0, 1.0, 3.0],
                0 => &[-2.0, 0.0, 2.0],
                _ => &[-1.5, -0.5, 0.5, 1.5],
            };
            assert_eq!(r, 2f64.powi(t.level as i32 - s.level as i32));
            assert!(c.iter().all(|x| allowed.contains(x)), "{s:?}: {c:?}");
        }
    }

    /// Stored coordinates of points that `points_to_morton` puts into a leaf lie in
    /// [−1 − β, 1 + β] with β of order ε₆₄ (|a| + w) / r_l (§3.13, "Containment"): about 2e-7 at level 16
    /// in the far domain. Error measure: |u| ≤ 1 + 1e-6.
    #[test]
    fn stored_coordinates_contained() {
        let mut rng = SplitMix64::new(3);
        for dom in domains(&mut rng) {
            for level in [0, 7, 16] {
                for key in targets(&mut rng, level) {
                    for k in 0..3 {
                        for u in dom.leaf_values(k, key, &[-1.0, 0.0, 1.0]) {
                            assert!(u.abs() <= 1.0 + 1e-6, "{}: {u}", dom.name);
                        }
                    }
                }
            }
        }
    }
}
