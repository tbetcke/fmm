//! The operator's P2P on every ISA (Helmholtz CH3.4c): `nd_fmm_simd::P2pKernel::
//! evaluate_helmholtz` against `P2pChoice::Reference` (`nd_fmm_ref::helmholtz::p2p::p2p`,
//! the trusted path), pair by pair (`PairOperator`), in f64 and f32, for real and complex k
//! with real and complex charges, with one vector (the source chunk's own strengths) and two
//! (gathered), with and without gradients.
//!
//! A target leaf of level 3 in the unit cube takes P2P from a neighbour of its level, from
//! a coarser neighbour of level 2 (r̂(s|t) = 2) and from itself, into one output chunk. Each
//! part of each target's potential and gradient lies within CH3.4b's per-term contract
//! (docs/design/simd-p2p.md §4.12) summed over the target's sources, with 2 n_s u_T per
//! term for accumulating n_s terms:
//!
//! - φ: Σ_j (32 + 12 |k̂| r_j + 2 n_s) u_T |q_j| e^{−Im k̂ r_j} / r_j;
//! - ∇φ: Σ_j (66 + 12 |k̂| r_j + 2 n_s) u_T |q_j| (1 + |k̂| r_j) e^{−Im k̂ r_j} / r_j²,
//!
//! in the target leaf's units (k̂ = k r_t). `Isa::Scalar` is the reference bit for bit, and
//! `Auto` is `Isa(Isa::detect())` bit for bit.

use mpi::traits::Equivalence;
use nd_fmm_exec::geometry::{self, Domain, radius};
use nd_fmm_exec::helmholtz::{ChargeKind, Charges, Densities, HelmholtzOperator, Wavenumber};
use nd_fmm_exec::operator::{Isa, IsaUnavailable, P2pChoice, SimdScalar};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

fn key(index: [usize; 3], level: usize) -> MortonKey {
    morton::from_index_and_level(index, level)
}

/// n points inside `key`, away from its faces by a twentieth of its half-width.
fn points_in(key: MortonKey, n: usize, rng: &mut Rng) -> Vec<[f64; 3]> {
    let domain = domain();
    let (c, r) = (
        geometry::centre(key, &domain),
        radius(morton::level(key) as usize, &domain),
    );
    (0..n)
        .map(|_| core::array::from_fn(|i| c[i] + 0.95 * r * rng.next()))
        .collect()
}

/// A charge part: a sign and a magnitude in [0.1, 1].
fn part(rng: &mut Rng) -> f64 {
    let x = rng.next();
    x.signum() * (0.1 + 0.9 * x.abs())
}

/// One source leaf: its key, points and the (Re q, Im q) of each vector, Im q zero for
/// real charges.
struct Leaf {
    key: MortonKey,
    points: Vec<[f64; 3]>,
    q: Vec<[Vec<f64>; 2]>,
}

impl Leaf {
    fn new(key: MortonKey, n: usize, kind: ChargeKind, m: usize, rng: &mut Rng) -> Self {
        let points = points_in(key, n, rng);
        let q = (0..m)
            .map(|_| {
                let re = (0..n).map(|_| part(rng)).collect();
                let im = match kind {
                    ChargeKind::Real => vec![0.0; n],
                    ChargeKind::Complex => (0..n).map(|_| part(rng)).collect(),
                };
                [re, im]
            })
            .collect();
        Self { key, points, q }
    }

    /// The source chunk in T: leaf-scaled coordinates, then the strengths.
    fn chunk<T: SimdScalar>(&self, d: &Densities, kind: ChargeKind) -> Vec<T> {
        let n = self.points.len();
        let mut chunk = vec![T::zero(); n * d.source_point_size()];
        for (u, &x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(&self.points) {
            *u = geometry::leaf_coordinates(x, self.key, &domain());
        }
        let planes: Vec<[Vec<T>; 2]> = self
            .q
            .iter()
            .map(|[re, im]| [re, im].map(|p| p.iter().map(|&x| T::from_f64(x)).collect()))
            .collect();
        let charges: Vec<Charges<'_, T>> = planes
            .iter()
            .map(|[re, im]| match kind {
                ChargeKind::Real => Charges::Real(re),
                ChargeKind::Complex => Charges::Complex { re, im },
            })
            .collect();
        let indices: Vec<usize> = (0..n).collect();
        d.write_leaf_charges(&charges, &indices, &mut chunk[3 * n..]);
        chunk
    }
}

/// The problem: a target leaf of level 3, its three source leaves (a neighbour of level
/// 3, a coarser neighbour of level 2, itself), k, the charge kind, m vectors and whether
/// the output holds gradients.
struct Problem {
    k: Wavenumber,
    kind: ChargeKind,
    m: usize,
    gradients: bool,
    densities: Densities,
    target: MortonKey,
    targets: Vec<[f64; 3]>,
    sources: Vec<Leaf>,
}

impl Problem {
    fn new(k: Wavenumber, kind: ChargeKind, m: usize, gradients: bool, seed: u64) -> Self {
        let mut rng = Rng(seed);
        // The level-3 leaf [0.375, 0.5] × [0.125, 0.25]²; the level-2 box [0.5, 0.75] ×
        // [0, 0.25]² shares its face x = 0.5.
        let target = key([3, 1, 1], 3);
        let own = Leaf::new(target, 13, kind, m, &mut rng);
        let targets = own.points.clone();
        let sources = vec![
            Leaf::new(key([3, 2, 1], 3), 20, kind, m, &mut rng),
            Leaf::new(key([2, 0, 0], 2), 23, kind, m, &mut rng),
            own,
        ];
        Self {
            k,
            kind,
            m,
            gradients,
            densities: Densities::new(k.kind(), kind, m),
            target,
            targets,
            sources,
        }
    }

    /// The operator of T with the P2P kernel of `choice`, degree 2 on every level (P2P
    /// does not read the degrees).
    fn operator<T: SimdScalar + Equivalence + Default>(
        &self,
        choice: P2pChoice,
    ) -> Result<HelmholtzOperator<T>, IsaUnavailable> {
        HelmholtzOperator::with_block_budget(
            self.k,
            self.densities,
            &domain(),
            &[2; 4],
            self.gradients,
            32,
            0,
        )
        .unwrap()
        .with_p2p(choice)
    }

    /// The target output chunk of P2P from the three source leaves, in that order.
    fn p2p<T: SimdScalar + Equivalence + Default>(&self, op: &mut HelmholtzOperator<T>) -> Vec<T> {
        let input: Vec<T> = self
            .targets
            .iter()
            .flat_map(|&x| geometry::leaf_coordinates::<T>(x, self.target, &domain()))
            .collect();
        let mut output = vec![T::zero(); self.targets.len() * op.target_output_point_size()];
        for leaf in &self.sources {
            let chunk = leaf.chunk::<T>(&self.densities, self.kind);
            op.p2p(leaf.key, self.target, &chunk, &input, &mut output);
        }
        output
    }

    /// The contract's bounds (module documentation) of target i and vector v, φ and ∇φ,
    /// in units of u_T.
    fn bounds(&self, i: usize, v: usize) -> (f64, f64) {
        let r_t = radius(3, &domain());
        let k_hat = (self.k.re() * r_t, self.k.im() * r_t);
        let k_abs = k_hat.0.hypot(k_hat.1);
        let n_s: usize = self.sources.iter().map(|l| l.points.len()).sum();
        let x = self.targets[i];
        let (mut phi, mut grad) = (0.0, 0.0);
        for leaf in &self.sources {
            for (j, y) in leaf.points.iter().enumerate() {
                let r = (0..3).map(|a| (x[a] - y[a]).powi(2)).sum::<f64>().sqrt() / r_t;
                if r == 0.0 {
                    continue;
                }
                let q = leaf.q[v][0][j].hypot(leaf.q[v][1][j]);
                let damped = q * (-k_hat.1 * r).exp();
                let accumulate = 2.0 * n_s as f64;
                phi += (32.0 + 12.0 * k_abs * r + accumulate) * damped / r;
                grad +=
                    (66.0 + 12.0 * k_abs * r + accumulate) * damped * (1.0 + k_abs * r) / (r * r);
            }
        }
        (phi, grad)
    }
}

/// The largest Re(k̂) r of the problem, Re(k) times the largest distance.
fn largest_phase(pb: &Problem) -> f64 {
    let mut far = 0.0f64;
    for x in &pb.targets {
        for leaf in &pb.sources {
            for y in &leaf.points {
                far = far.max((0..3).map(|a| (x[a] - y[a]).powi(2)).sum::<f64>().sqrt());
            }
        }
    }
    pb.k.re() * far
}

/// Every available ISA against the reference within the contract, `Isa::Scalar` and
/// `Auto` bit for bit; the worst ratio of difference to bound printed per ISA.
fn every_isa<T: SimdScalar + Equivalence + Default>(pb: &Problem) {
    let u = T::epsilon().to_f64() / 2.0;
    let mut reference_op = pb.operator::<T>(P2pChoice::Reference).unwrap();
    assert_eq!(reference_op.p2p_kernel(), P2pChoice::Reference);
    let reference = pb.p2p(&mut reference_op);
    let d = &pb.densities;
    let n = pb.targets.len();
    let name = format!(
        "{} k = {} + {}i, {:?} charges, m = {}, gradients {}",
        core::any::type_name::<T>(),
        pb.k.re(),
        pb.k.im(),
        pb.kind,
        pb.m,
        pb.gradients
    );
    for isa in Isa::all() {
        let Ok(mut op) = pb.operator::<T>(P2pChoice::Isa(isa)) else {
            assert!(!isa.is_available(), "{isa} is available but refused");
            assert_eq!(
                pb.operator::<T>(P2pChoice::Isa(isa)).err(),
                Some(IsaUnavailable { isa })
            );
            continue;
        };
        assert_eq!(op.p2p_kernel(), P2pChoice::Isa(isa));
        let output = pb.p2p(&mut op);
        if isa == Isa::Scalar {
            assert!(
                output == reference,
                "{name}: Isa::Scalar is not the reference"
            );
        }
        let mut worst = (0.0f64, 0.0f64);
        for v in 0..pb.m {
            for i in 0..n {
                let (phi, grad) = pb.bounds(i, v);
                for part in 0..2 {
                    let o = d.output_column(v, part);
                    let index = d.potential_index(n, o, i);
                    let diff = (output[index].to_f64() - reference[index].to_f64()).abs();
                    worst.0 = worst.0.max(diff / (phi * u));
                    assert!(
                        diff <= phi * u,
                        "{name}, {isa}: φ of target {i}, vector {v}, part {part}: {:e} \
                         against {:e}, bound {:e}",
                        output[index].to_f64(),
                        reference[index].to_f64(),
                        phi * u
                    );
                    for axis in (0..3).filter(|_| pb.gradients) {
                        let index = d.gradient_index(n, o, i, axis);
                        let diff = (output[index].to_f64() - reference[index].to_f64()).abs();
                        worst.1 = worst.1.max(diff / (grad * u));
                        assert!(
                            diff <= grad * u,
                            "{name}, {isa}: ∇φ of target {i}, vector {v}, part {part}, axis \
                             {axis}: {:e} against {:e}, bound {:e}",
                            output[index].to_f64(),
                            reference[index].to_f64(),
                            grad * u
                        );
                    }
                }
            }
        }
        println!(
            "{name}, {isa}: worst difference / bound φ {:.3}, ∇φ {:.3}",
            worst.0, worst.1
        );
    }
    let mut auto = pb.operator::<T>(P2pChoice::Auto).unwrap();
    assert_eq!(auto.p2p_kernel(), P2pChoice::Isa(Isa::detect()));
    let mut detected = pb.operator::<T>(P2pChoice::Isa(Isa::detect())).unwrap();
    assert!(
        pb.p2p(&mut auto) == pb.p2p(&mut detected),
        "{name}: Auto is not Isa::detect()"
    );
}

/// Real k (Re k̂ r to about 53, beyond the leaf rule's 22 c_leaf at c_leaf = 1) and complex
/// k, real and complex charges, f64 and f32; two vectors with gradients, one vector with
/// and without.
#[test]
fn every_isa_within_the_contract() {
    for (re, im) in [(1.0, 0.0), (30.0, 3.0), (128.0, 0.0), (128.0, 16.0)] {
        for kind in [ChargeKind::Real, ChargeKind::Complex] {
            for (m, gradients) in [(2, true), (1, true), (1, false)] {
                let k = Wavenumber::new(re, im).unwrap();
                let pb = Problem::new(k, kind, m, gradients, 17);
                println!("k = {re} + {im}i: largest Re k̂ r {:.1}", largest_phase(&pb));
                every_isa::<f64>(&pb);
                every_isa::<f32>(&pb);
            }
        }
    }
}

/// The default kernel is `Auto`'s.
#[test]
fn the_default_is_auto() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let d = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let op = HelmholtzOperator::<f64>::new(k, d, &domain(), &[2; 4], false, 8).unwrap();
    assert_eq!(op.p2p_kernel(), P2pChoice::Isa(Isa::detect()));
}
