//! Where the Helmholtz FMM's errors concentrate (Helmholtz CH4.0b;
//! docs/design/helmholtz-fmm-proposal.md §9.1, the box measure). MPI-free.
//!
//! - **The reference.** The direct sum in f64 (`nd_fmm_ref::helmholtz::p2p::direct_sum`,
//!   with 1/(4π)) at a set of targets: every point ([`Reference::every`]), or §9.1's samples
//!   with up to s targets per box of a level ([`box_sample`], [`Reference::at`]), the
//!   "sampled direct sum" of the roadmap.
//! - **The errors.** Per reference target the complex errors of φ and of ∇φ's three
//!   components ([`Errors`]), from an output at every point in the problem's order.
//! - **The measures.** The relative ℓ2 errors over the reference's targets, and §9.1's box
//!   measure: per box of a [`Grid`], the largest error at one of its targets over the
//!   reference's RMS over its targets ([`Errors::boxes`], as
//!   [`crosscheck`](crate::helmholtz_fmm3d::crosscheck) with the reference at every point).
//! - **The place.** A target's offset from the centre of its box on each level, in units of
//!   the box's half-width ([`offsets`]): ±1 on a face.
//! - **The levels.** The error of one run minus that of a run with one level's degree raised
//!   is that level's part of the error ([`level_shares`]): with the levels' parts adding up,
//!   the errors of the run are their sum plus a rest.

use std::collections::HashMap;
use std::thread;

use nd_fmm_exec::helmholtz::Output;
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p::direct_sum;

use crate::SplitMix64;
use crate::helmholtz_accuracy::Problem;
use crate::helmholtz_fmm3d::Grid;

/// The direct sum at a set of targets for one charge vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    /// The targets' positions in the problem's points, ascending.
    pub targets: Vec<usize>,
    /// φ at each target, with 1/(4π).
    pub potential: Vec<Complex<f64>>,
    /// ∇φ at each target, with 1/(4π).
    pub gradient: Vec<[Complex<f64>; 3]>,
}

impl Reference {
    /// The reference at every point of `problem`.
    pub fn every(k: Complex<f64>, problem: &Problem, vector: usize, threads: usize) -> Self {
        Self::at(k, problem, vector, (0..problem.len()).collect(), threads)
    }

    /// The reference at the points `targets` (positions, sorted and made distinct) on
    /// `threads` threads, each a contiguous share of the targets, so the values do not
    /// depend on the thread count; each value is
    /// [`Oracle::of_samples`](crate::helmholtz_accuracy::Oracle::of_samples)'s at the same
    /// point bit for bit.
    pub fn at(
        k: Complex<f64>,
        problem: &Problem,
        vector: usize,
        mut targets: Vec<usize>,
        threads: usize,
    ) -> Self {
        targets.sort_unstable();
        targets.dedup();
        let points: Vec<[f64; 3]> = targets.iter().map(|&i| problem.points[i]).collect();
        let charges = problem.complex_charges(vector);
        let zero = Complex::new(0.0, 0.0);
        let mut potential = vec![zero; points.len()];
        let mut gradient = vec![[zero; 3]; points.len()];
        let chunk = points.len().div_ceil(threads.max(1)).max(1);
        thread::scope(|scope| {
            for ((x, phi), grad) in points
                .chunks(chunk)
                .zip(potential.chunks_mut(chunk))
                .zip(gradient.chunks_mut(chunk))
            {
                let (sources, charges) = (&problem.points, &charges);
                scope.spawn(move || direct_sum(k, sources, charges, x, phi, Some(grad)));
            }
        });
        let scale = 1.0 / (4.0 * std::f64::consts::PI);
        Self {
            targets,
            potential: potential.into_iter().map(|v| v * scale).collect(),
            gradient: gradient.into_iter().map(|g| g.map(|v| v * scale)).collect(),
        }
    }

    /// The RMS over the targets of |φ| and of |∇φ|.
    pub fn rms(&self) -> [f64; 2] {
        let n = self.targets.len().max(1) as f64;
        let phi: f64 = self.potential.iter().map(|v| v.norm_sqr()).sum();
        let grad: f64 = self
            .gradient
            .iter()
            .flat_map(|g| g.iter().map(|v| v.norm_sqr()))
            .sum();
        [(phi / n).sqrt(), (grad / n).sqrt()]
    }
}

/// The targets of the sampled reference: `sample` (§9.1's), and in every box of `grid` that
/// holds points up to `per_box` of them, drawn without replacement from the generator of
/// `seed` box by box in key order (all of a box's points where it holds at most
/// `per_box`). Sorted and distinct.
pub fn box_sample(
    points: &[[f64; 3]],
    sample: &[usize],
    grid: &Grid,
    per_box: usize,
    seed: u64,
) -> Vec<usize> {
    let mut boxes: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &x) in points.iter().enumerate() {
        boxes.entry(grid.box_of(x)).or_default().push(i);
    }
    let mut keys: Vec<u64> = boxes.keys().copied().collect();
    keys.sort_unstable();
    let mut rng = SplitMix64::new(seed);
    let mut targets: Vec<usize> = sample.to_vec();
    for key in keys {
        let mut members = boxes.remove(&key).expect("a box of the keys");
        let take = per_box.min(members.len());
        for i in 0..take {
            let n = members.len() - i;
            let j = i + ((((rng.next_u64() >> 32) * n as u64) >> 32) as usize);
            members.swap(i, j);
        }
        targets.extend_from_slice(&members[..take]);
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// The errors of an output at a reference's targets: per target [Δφ, Δ∂x, Δ∂y, Δ∂z].
#[derive(Clone, Debug, PartialEq)]
pub struct Errors {
    /// Per reference target, the complex errors of φ and of ∇φ's components.
    pub values: Vec<[Complex<f64>; 4]>,
}

/// One box of the box measure.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxError {
    /// The box (`Grid::box_of`'s key).
    pub key: u64,
    /// The largest error [φ, ∇φ] at one of its targets over the reference's RMS.
    pub error: [f64; 2],
    /// The index into the reference's targets of the target of each largest error.
    pub worst: [usize; 2],
    /// The box's reference targets.
    pub targets: usize,
}

impl Errors {
    /// The errors of vector `vector` of `output`, the FMM's output at every point of the
    /// problem in its order (one rank), at `reference`'s targets.
    ///
    /// # Panics
    ///
    /// If `output` has no gradients or fewer targets than a reference target's position.
    pub fn of(output: &Output<f64>, vector: usize, reference: &Reference) -> Self {
        let phi = output.potential_of(vector);
        let grad = output.gradient_of(vector).expect("gradients");
        let values = reference
            .targets
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let p = Complex::new(phi.re[i], phi.im.map_or(0.0, |im| im[i]));
                let g: [Complex<f64>; 3] = std::array::from_fn(|a| {
                    Complex::new(grad.re[i][a], grad.im.map_or(0.0, |im| im[i][a]))
                });
                let (rp, rg) = (reference.potential[j], reference.gradient[j]);
                [p - rp, g[0] - rg[0], g[1] - rg[1], g[2] - rg[2]]
            })
            .collect();
        Self { values }
    }

    /// |Δφ| and |Δ∇φ| at target j.
    pub fn norms(&self, j: usize) -> [f64; 2] {
        let e = &self.values[j];
        // As `crosscheck`: the square roots of the squared norms.
        [
            e[0].norm_sqr().sqrt(),
            (e[1].norm_sqr() + e[2].norm_sqr() + e[3].norm_sqr()).sqrt(),
        ]
    }

    /// The relative ℓ2 errors [φ, ∇φ] over the reference's targets.
    pub fn l2(&self, reference: &Reference) -> [f64; 2] {
        let n = self.values.len().max(1) as f64;
        let rms = reference.rms();
        let sums = (0..self.values.len()).fold([0.0f64; 2], |s, j| {
            let e = self.norms(j);
            [s[0] + e[0] * e[0], s[1] + e[1] * e[1]]
        });
        let ratio = |e: f64, r: f64| if r > 0.0 { (e / n).sqrt() / r } else { 0.0 };
        [ratio(sums[0], rms[0]), ratio(sums[1], rms[1])]
    }

    /// §9.1's box measure on `grid`: every box with a reference target, sorted by key.
    pub fn boxes(&self, reference: &Reference, points: &[[f64; 3]], grid: &Grid) -> Vec<BoxError> {
        let rms = reference.rms();
        let scale = |e: f64, r: f64| if r > 0.0 { e / r } else { 0.0 };
        let mut boxes: HashMap<u64, BoxError> = HashMap::new();
        for (j, &i) in reference.targets.iter().enumerate() {
            let key = grid.box_of(points[i]);
            let e = self.norms(j);
            let entry = boxes.entry(key).or_insert(BoxError {
                key,
                error: [0.0; 2],
                worst: [j; 2],
                targets: 0,
            });
            entry.targets += 1;
            for f in 0..2 {
                let e = scale(e[f], rms[f]);
                if e > entry.error[f] {
                    entry.error[f] = e;
                    entry.worst[f] = j;
                }
            }
        }
        let mut boxes: Vec<BoxError> = boxes.into_values().collect();
        boxes.sort_by_key(|b| b.key);
        boxes
    }
}

/// The boxes of `boxes` with the `count` largest errors of field `field` (0 φ, 1 ∇φ),
/// largest first, ties by key.
pub fn worst_boxes(boxes: &[BoxError], field: usize, count: usize) -> Vec<BoxError> {
    let mut sorted = boxes.to_vec();
    sorted.sort_by(|a, b| {
        b.error[field]
            .total_cmp(&a.error[field])
            .then(a.key.cmp(&b.key))
    });
    sorted.truncate(count);
    sorted
}

/// The (i, j, k) of a [`Grid`] key.
pub fn cell(key: u64) -> [u64; 3] {
    let mask = (1u64 << 21) - 1;
    [key & mask, (key >> 21) & mask, (key >> 42) & mask]
}

/// The offset of `x` from the centre of its box on each level 0..=`deepest` of `grid`'s
/// domain, per axis in units of the box's half-width: in [−1, 1], ±1 on a face.
pub fn offsets(x: [f64; 3], grid: &Grid, deepest: usize) -> Vec<[f64; 3]> {
    let (lower, side) = (grid.domain.lower(), grid.domain.side());
    (0..=deepest)
        .map(|l| {
            let at = Grid {
                domain: grid.domain,
                level: l,
            };
            let c = cell(at.box_of(x));
            let width = side / (1u64 << l) as f64;
            std::array::from_fn(|a| {
                let centre = lower[a] + (c[a] as f64 + 0.5) * width;
                (x[a] - centre) / (0.5 * width)
            })
        })
        .collect()
}

/// The largest |offset| over the axes, per level, of [`offsets`].
pub fn face_closeness(offsets: &[[f64; 3]]) -> Vec<f64> {
    offsets
        .iter()
        .map(|o| o.iter().fold(0.0f64, |m, v| m.max(v.abs())))
        .collect()
}

/// The parts of the errors `base` by level, from runs with one level each raised
/// (`raised[l]`, `None` for a level not run): level l's part at a target is base − raised,
/// and per field [φ, ∇φ] each level's share Σ|part|² / Σ|base|², and last the rest's,
/// |base − Σ parts|² / Σ|base|².
///
/// # Panics
///
/// If an entry of `raised` has another number of targets than `base`.
pub fn level_shares(base: &Errors, raised: &[Option<Errors>]) -> Vec<[f64; 2]> {
    let n = base.values.len();
    let total = (0..n).fold([0.0f64; 2], |s, j| {
        let e = base.norms(j);
        [s[0] + e[0] * e[0], s[1] + e[1] * e[1]]
    });
    let mut rest = base.values.clone();
    let mut shares = Vec::with_capacity(raised.len() + 1);
    let share = |s: f64, t: f64| if t > 0.0 { s / t } else { 0.0 };
    for errors in raised {
        let Some(errors) = errors else {
            shares.push([0.0; 2]);
            continue;
        };
        assert_eq!(errors.values.len(), n, "the same targets");
        let mut sums = [0.0f64; 2];
        for (j, r) in rest.iter_mut().enumerate() {
            let part: [Complex<f64>; 4] =
                std::array::from_fn(|c| base.values[j][c] - errors.values[j][c]);
            sums[0] += part[0].norm_sqr();
            sums[1] += part[1..].iter().map(|v| v.norm_sqr()).sum::<f64>();
            for c in 0..4 {
                r[c] -= part[c];
            }
        }
        shares.push([share(sums[0], total[0]), share(sums[1], total[1])]);
    }
    let rest = Errors { values: rest };
    let sums = (0..n).fold([0.0f64; 2], |s, j| {
        let e = rest.norms(j);
        [s[0] + e[0] * e[0], s[1] + e[1] * e[1]]
    });
    shares.push([share(sums[0], total[0]), share(sums[1], total[1])]);
    shares
}

/// Level l's part of the error at reference target j, |base − raised\[l\]| per field, over
/// |base| at j.
pub fn level_parts_at(base: &Errors, raised: &[Option<Errors>], j: usize) -> Vec<[f64; 2]> {
    let b = base.norms(j);
    raised
        .iter()
        .map(|errors| {
            errors.as_ref().map_or([0.0; 2], |errors| {
                let part = Errors {
                    values: vec![std::array::from_fn(|c| {
                        base.values[j][c] - errors.values[j][c]
                    })],
                };
                let e = part.norms(0);
                let ratio = |e: f64, b: f64| if b > 0.0 { e / b } else { 0.0 };
                [ratio(e[0], b[0]), ratio(e[1], b[1])]
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nd_fmm_exec::geometry::Domain;
    use nd_octree::PhysicalBox;

    fn unit_grid(level: usize) -> Grid {
        Grid {
            domain: Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
                .expect("a domain"),
            level,
        }
    }

    #[test]
    fn offsets_on_a_face_and_at_a_centre() {
        let grid = unit_grid(3);
        // On the domain's upper z face at every level.
        let top = offsets([0.3, 0.6, 1.0], &grid, 3);
        for o in &top {
            assert!((o[2] - 1.0).abs() < 1e-12, "{o:?}");
        }
        // The centre of the level-2 box (1, 2, 3) is (0.375, 0.625, 0.875).
        let centre = offsets([0.375, 0.625, 0.875], &grid, 2);
        assert_eq!(centre[2], [0.0; 3]);
        assert_eq!(face_closeness(&centre)[2], 0.0);
        // Level 0: the offset from the root's centre in units of 0.5.
        assert!((centre[0][0] - (-0.25)).abs() < 1e-15);
    }

    #[test]
    fn box_sample_takes_up_to_s_per_box() {
        let mut rng = SplitMix64::new(3);
        let points: Vec<[f64; 3]> = (0..500)
            .map(|_| {
                [
                    rng.range(0.0, 1.0),
                    rng.range(0.0, 1.0),
                    rng.range(0.0, 0.25),
                ]
            })
            .collect();
        let grid = unit_grid(2);
        let targets = box_sample(&points, &[7, 3], &grid, 4, 11);
        assert_eq!(
            targets,
            box_sample(&points, &[3, 7], &grid, 4, 11),
            "deterministic"
        );
        assert!(targets.windows(2).all(|w| w[0] < w[1]), "sorted, distinct");
        assert!(
            targets.contains(&7) && targets.contains(&3),
            "the samples kept"
        );
        let mut counts: HashMap<u64, (usize, usize)> = HashMap::new();
        for (i, &x) in points.iter().enumerate() {
            let entry = counts.entry(grid.box_of(x)).or_default();
            entry.0 += 1;
            if targets.binary_search(&i).is_ok() && i != 3 && i != 7 {
                entry.1 += 1;
            }
        }
        for (held, taken) in counts.values() {
            assert!(
                *taken <= 4 && *taken >= (*held).min(4).saturating_sub(2),
                "{held} {taken}"
            );
        }
        // A box with fewer points than s gives all of them.
        let all = box_sample(&points[..3], &[], &unit_grid(0), 10, 1);
        assert_eq!(all, vec![0, 1, 2]);
    }

    #[test]
    fn level_shares_add_up() {
        let c = |re: f64| Complex::new(re, 0.0);
        let zero = c(0.0);
        let base = Errors {
            values: vec![[c(3.0), c(4.0), zero, zero], [c(1.0), zero, zero, zero]],
        };
        // Level 0 holds the first target's φ error, level 2 its ∇φ error; the rest is the
        // second target's φ error.
        let raised = vec![
            Some(Errors {
                values: vec![[zero, c(4.0), zero, zero], [c(1.0), zero, zero, zero]],
            }),
            None,
            Some(Errors {
                values: vec![[c(3.0), zero, zero, zero], [c(1.0), zero, zero, zero]],
            }),
        ];
        let shares = level_shares(&base, &raised);
        assert_eq!(shares.len(), 4);
        assert_eq!(shares[0], [0.9, 0.0]);
        assert_eq!(shares[1], [0.0, 0.0]);
        assert_eq!(shares[2], [0.0, 1.0]);
        assert_eq!(shares[3], [0.1, 0.0]);
        let at = level_parts_at(&base, &raised, 0);
        assert_eq!(at[0], [1.0, 0.0]);
        assert_eq!(at[2], [0.0, 1.0]);
    }

    #[test]
    fn worst_boxes_largest_first() {
        let b = |key, e: f64| BoxError {
            key,
            error: [e, 2.0 * e],
            worst: [0; 2],
            targets: 1,
        };
        let boxes = [b(1, 0.5), b(2, 3.0), b(3, 1.0), b(4, 3.0)];
        let worst = worst_boxes(&boxes, 1, 3);
        assert_eq!(
            worst.iter().map(|b| b.key).collect::<Vec<_>>(),
            vec![2, 4, 3]
        );
        assert_eq!(cell(5 | (6 << 21) | (7 << 42)), [5, 6, 7]);
    }
}
