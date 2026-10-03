//! The leaf-scaled pairs of the §3.13 check on the host: adversarial pairs and seeded
//! random pairs on levels 0–16 (`tools/fixtures/check_p2p_domain.py`, rebuilt from its
//! description), with the values a correct kernel must compute.

use std::collections::HashMap;

use nd_fmm_validate::SplitMix64;

use crate::geometry::{
    DEEPEST_LEVEL, Domain, Key, SELF_POSITIONS, domains, locus, relative_frame, targets, u_list,
};
use crate::real::Real;

/// One source–target pair in leaf-scaled coordinates, as a P2P kernel receives it.
#[derive(Clone, Copy, Debug)]
pub struct Pair<T> {
    /// u_t.
    pub ut: [T; 3],
    /// u_s, in the frame of its own leaf.
    pub us: [T; 3],
    /// ĉ(s|t).
    pub c: [T; 3],
    /// r̂(s|t).
    pub r: T,
    /// s = t: ŷ = u_s, no arithmetic.
    pub self_pair: bool,
    /// Whether more than one double point gives these stored values (a coincidence here
    /// can come from distinct points, §3.13 "Coincident pairs").
    pub multi: bool,
}

impl<T: Real> Pair<T> {
    /// ŷ = fl(ĉ + r̂ u_s) (u_s for s = t), on the host. r̂ u_s is exact.
    pub fn y(&self) -> [T; 3] {
        if self.self_pair {
            self.us
        } else {
            std::array::from_fn(|k| self.c[k] + self.r * self.us[k])
        }
    }

    /// dₖ = fl(u_t,k − ŷₖ), on the host.
    pub fn d(&self) -> [T; 3] {
        let y = self.y();
        std::array::from_fn(|k| self.ut[k] - y[k])
    }

    /// Whether the reference skips the pair: u_t == ŷ in all three components.
    pub fn coincident(&self) -> bool {
        let y = self.y();
        (0..3).all(|k| self.ut[k] == y[k])
    }

    /// r² as the reference forms it, ((d₀² + d₁²) + d₂²), unfused, on the host.
    pub fn r2_plain(&self) -> T {
        let d = self.d();
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }

    /// r² = fma(d₂, d₂, fma(d₁, d₁, d₀²)), on the host.
    pub fn r2_forward(&self) -> T {
        let d = self.d();
        T::host_fma(d[2], d[2], T::host_fma(d[1], d[1], d[0] * d[0]))
    }

    /// r² = fma(d₀, d₀, fma(d₁, d₁, d₂²)), on the host.
    pub fn r2_backward(&self) -> T {
        let d = self.d();
        T::host_fma(d[0], d[0], T::host_fma(d[1], d[1], d[2] * d[2]))
    }
}

/// One component of a pair: (d, u_t, u_s, more than one point).
#[derive(Clone, Copy)]
struct Entry<T> {
    d: T,
    ut: T,
    us: T,
    multi: bool,
}

/// The distinct stored values in T of a list of f64 stored values, each with the number of
/// points that give it.
fn distinct<T: Real>(values: &[f64]) -> Vec<(T, usize)> {
    let mut out: Vec<(T, usize)> = Vec::new();
    for &u in values {
        let v = T::narrow(u);
        match out.iter_mut().find(|(w, _)| w.bits() == v.bits()) {
            Some(slot) => slot.1 += 1,
            None => out.push((v, 1)),
        }
    }
    out
}

type AxisKey = (usize, usize, u32, u32, u32, u32);

/// The component entries of the pair (s, t) along axis k, sorted by |d|.
fn axis_entries<T: Real>(
    cache: &mut HashMap<AxisKey, Vec<Entry<T>>>,
    dom: (usize, &Domain),
    k: usize,
    s: Key,
    t: Key,
) -> Vec<Entry<T>> {
    let key = (dom.0, k, s.level, s.index[k], t.level, t.index[k]);
    if let Some(e) = cache.get(&key) {
        return e.clone();
    }
    let self_pair = s == t;
    let (c, r) = relative_frame(s, t);
    let (c_k, r) = (c[k], r);
    let (pos_t, pos_s): (Vec<f64>, Vec<f64>) = if self_pair {
        (SELF_POSITIONS.to_vec(), SELF_POSITIONS.to_vec())
    } else {
        let pt = locus(c_k, r);
        let ps = pt.iter().map(|p| (p - c_k) / r).collect();
        (pt, ps)
    };
    let tv = distinct::<T>(&dom.1.leaf_values(k, t, &pos_t));
    let sv = if self_pair {
        tv.clone()
    } else {
        distinct::<T>(&dom.1.leaf_values(k, s, &pos_s))
    };
    let (ct, rt) = (T::narrow(c_k), T::narrow(r));
    let mut out = Vec::with_capacity(tv.len() * sv.len());
    for &(us, ns) in &sv {
        let y = if self_pair { us } else { ct + rt * us };
        for &(ut, _) in &tv {
            out.push(Entry {
                d: ut - y,
                ut,
                us,
                // Equal values from distinct points: always for s ≠ t (different
                // leaves), and for s = t when several points give the value.
                multi: !self_pair || (ut.bits() == us.bits() && ns > 1),
            });
        }
    }
    out.sort_by(|a, b| a.d.widen().abs().total_cmp(&b.d.widen().abs()));
    cache.insert(key, out.clone());
    out
}

/// The adversarial pairs: on every domain and level, target leaves at the first, last,
/// middle and random indices, the leaf itself and every U-list neighbour 2:1 balance
/// allows, with points within a few ulps of the leaf centres, the faces, the source
/// centres and the cancellations ĉ + r̂ u_s ≈ 0. Per component the entries with d = 0 (if
/// any), the smallest nonzero |d| and the largest are combined, as the script does.
///
/// `levels` limits the levels (the smoke test runs a few).
pub fn adversarial<T: Real>(seed: u64, levels: &[u32]) -> Vec<Pair<T>> {
    let mut rng = SplitMix64::new(seed);
    let doms = domains(&mut rng);
    let mut cache = HashMap::new();
    let mut out = Vec::new();
    for (di, dom) in doms.iter().enumerate() {
        for &level in levels {
            for t in targets(&mut rng, level) {
                for s in std::iter::once(t).chain(u_list(t)) {
                    let per_axis: Vec<Vec<Entry<T>>> = (0..3)
                        .map(|k| axis_entries(&mut cache, (di, dom), k, s, t))
                        .collect();
                    if per_axis.iter().any(Vec::is_empty) {
                        continue;
                    }
                    let choices: Vec<Vec<Entry<T>>> = per_axis
                        .iter()
                        .map(|entries| {
                            let mut pick: Vec<Entry<T>> = entries
                                .iter()
                                .filter(|e| e.d.widen() == 0.0)
                                .take(1)
                                .copied()
                                .collect();
                            let nonzero: Vec<Entry<T>> = entries
                                .iter()
                                .filter(|e| e.d.widen() != 0.0)
                                .copied()
                                .collect();
                            pick.extend(nonzero.first());
                            pick.extend(nonzero.last());
                            pick
                        })
                        .collect();
                    let (c, r) = relative_frame(s, t);
                    for a in &choices[0] {
                        for b in &choices[1] {
                            for e in &choices[2] {
                                out.push(Pair {
                                    ut: [a.ut, b.ut, e.ut],
                                    us: [a.us, b.us, e.us],
                                    c: c.map(T::narrow),
                                    r: T::narrow(r),
                                    self_pair: s == t,
                                    multi: a.multi || b.multi || e.multi,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Every level 0–16.
pub fn all_levels() -> Vec<u32> {
    (0..=DEEPEST_LEVEL).collect()
}

/// Seeded random pairs on levels 0–16 in the dyadic, generic and far domains: `trials`
/// (target, source) leaf pairs per domain, the source the leaf itself with probability ¼
/// or a random U-list neighbour, six points in each, the first two duplicated in self
/// pairs (coincident pairs).
pub fn random<T: Real>(seed: u64, trials: usize) -> Vec<Pair<T>> {
    let mut rng = SplitMix64::new(seed);
    let doms = domains(&mut rng);
    let mut out = Vec::new();
    for dom in doms.iter().skip(1) {
        for _ in 0..trials {
            let level = (rng.next_u64() % u64::from(DEEPEST_LEVEL + 1)) as u32;
            let n = 1u64 << level;
            let t = Key {
                level,
                index: std::array::from_fn(|_| (rng.next_u64() % n) as u32),
            };
            let neighbours = u_list(t);
            let s = if neighbours.is_empty() || rng.uniform() < 0.25 {
                t
            } else {
                neighbours[(rng.next_u64() % neighbours.len() as u64) as usize]
            };
            let xt: Vec<[f64; 3]> = (0..6).map(|_| dom.random_point(&mut rng, t)).collect();
            let mut xs: Vec<[f64; 3]> = (0..6).map(|_| dom.random_point(&mut rng, s)).collect();
            if s == t {
                xs[..2].copy_from_slice(&xt[..2]);
            }
            let (c, r) = relative_frame(s, t);
            for x in &xt {
                let ut = dom.stored(*x, t).map(T::narrow);
                for y in &xs {
                    out.push(Pair {
                        ut,
                        us: dom.stored(*y, s).map(T::narrow),
                        c: c.map(T::narrow),
                        r: T::narrow(r),
                        self_pair: s == t,
                        multi: false,
                    });
                }
            }
        }
    }
    out
}
