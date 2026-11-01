//! Complex densities (Helmholtz CH3.0c; CONVENTIONS §3.15, "Storage" and "Inputs and
//! outputs"): the sizes of every path against §3.15's table, the chunk indices against
//! §3.13's several-vector layout, the charges in, P2M and L2P bit for bit against
//! `nd_fmm_ref::helmholtz::leaf`, Yukawa's complex vectors as two real ones, and the
//! leaf's chunks end to end against the direct sum (`nd_fmm_ref::helmholtz::p2p::
//! direct_sum`, 1/(4π) applied), relative L2 over the targets.

use std::f64::consts::PI;

use nd_fmm_exec::geometry::{self, Domain};
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, Densities, DensityError, LeafScratch, Output, Wavenumber, WavenumberKind,
};
use nd_fmm_math::Complex;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, p2p};
use nd_octree::{MortonKey, PhysicalBox, morton};

type C = Complex<f64>;

const KINDS: [WavenumberKind; 3] = [
    WavenumberKind::Real,
    WavenumberKind::Complex,
    WavenumberKind::Yukawa,
];
const CHARGES: [ChargeKind; 2] = [ChargeKind::Real, ChargeKind::Complex];

/// SplitMix64, uniform in [−1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

/// A wavenumber of `kind` with |k| about `size`.
fn wavenumber(kind: WavenumberKind, size: f64) -> Wavenumber {
    match kind {
        WavenumberKind::Real => Wavenumber::new(size, 0.0),
        WavenumberKind::Complex => Wavenumber::new(size, 0.3 * size),
        WavenumberKind::Yukawa => Wavenumber::new(0.0, size),
    }
    .unwrap()
}

/// The domain [−0.3, 1.7]³ (side 2) and a leaf on level 3 of it (half-width 0.125).
fn leaf_box() -> (Domain, MortonKey) {
    let domain = Domain::new(&PhysicalBox::new([-0.3, -0.3, -0.3, 1.7, 1.7, 1.7])).unwrap();
    (domain, morton::from_index_and_level([3, 5, 2], 3))
}

/// The m charge vectors of `kind` at n sources, as planes: (Re q, Im q) per vector, Im q
/// zero for real charges.
fn charge_planes(kind: ChargeKind, m: usize, n: usize, rng: &mut Rng) -> Vec<[Vec<f64>; 2]> {
    (0..m)
        .map(|_| {
            let re = (0..n).map(|_| rng.next()).collect();
            let im = match kind {
                ChargeKind::Real => vec![0.0; n],
                ChargeKind::Complex => (0..n).map(|_| rng.next()).collect(),
            };
            [re, im]
        })
        .collect()
}

/// `planes` as [`Charges`] of `kind`.
fn as_charges(kind: ChargeKind, planes: &[[Vec<f64>; 2]]) -> Vec<Charges<'_, f64>> {
    planes
        .iter()
        .map(|[re, im]| match kind {
            ChargeKind::Real => Charges::Real(re),
            ChargeKind::Complex => Charges::Complex { re, im },
        })
        .collect()
}

/// Vector v of `planes` as complex charges.
fn complex_of(planes: &[[Vec<f64>; 2]], v: usize) -> Vec<C> {
    let [re, im] = &planes[v];
    re.iter().zip(im).map(|(&a, &b)| C::new(a, b)).collect()
}

/// The source chunk of the leaf for absolute points `points` and the charges `planes`
/// (the leaf's point j is source j).
fn source_chunk(
    densities: &Densities,
    domain: &Domain,
    key: MortonKey,
    points: &[[f64; 3]],
    planes: &[[Vec<f64>; 2]],
) -> Vec<f64> {
    let n = points.len();
    let mut chunk = vec![f64::NAN; n * densities.source_point_size()];
    for (u, &x) in chunk[..3 * n].as_chunks_mut::<3>().0.iter_mut().zip(points) {
        *u = geometry::leaf_coordinates::<f64>(x, key, domain);
    }
    let indices: Vec<usize> = (0..n).collect();
    let charges = as_charges(densities.charges(), planes);
    densities.write_leaf_charges(&charges, &indices, &mut chunk[3 * n..]);
    chunk
}

/// n points inside the leaf, absolute.
fn leaf_points(domain: &Domain, key: MortonKey, n: usize, rng: &mut Rng) -> Vec<[f64; 3]> {
    let c = geometry::centre(key, domain);
    let r = geometry::radius(morton::level(key) as usize, domain);
    (0..n)
        .map(|_| core::array::from_fn(|a| c[a] + 0.999 * r * rng.next()))
        .collect()
}

/// n points at distance `distance` × the leaf's circumradius from its centre, absolute.
fn far_points(
    domain: &Domain,
    key: MortonKey,
    n: usize,
    distance: f64,
    rng: &mut Rng,
) -> Vec<[f64; 3]> {
    let c = geometry::centre(key, domain);
    let a = 3f64.sqrt() * geometry::radius(morton::level(key) as usize, domain);
    (0..n)
        .map(|_| {
            let d: [f64; 3] = core::array::from_fn(|_| rng.next());
            let norm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            core::array::from_fn(|i| c[i] + distance * a * d[i] / norm)
        })
        .collect()
}

/// The leaf's level and frame in its own units, as the library forms them.
fn leaf_level(k: &Wavenumber, r: f64) -> (Level<f64>, Frame<f64>) {
    (
        Level::new(C::new(k.re() * r, k.im() * r), 1.0),
        Frame::new([0.0; 3], 1.0),
    )
}

/// The leaf-unit coordinates of `chunk`'s n points.
fn chunk_points(chunk: &[f64], n: usize) -> Vec<[f64; 3]> {
    chunk[..3 * n].as_chunks::<3>().0.to_vec()
}

/// The relative L2 error of `a` against `b`.
fn relative_l2(a: &[C], b: &[C]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
    let norm: f64 = b.iter().map(C::norm_sqr).sum();
    (diff / norm).sqrt()
}

/// φ and ∇φ of the kernel G_k (1/(4π) included) at `targets`, complex k = Re k + i Im k.
fn exact(
    k: &Wavenumber,
    sources: &[[f64; 3]],
    charges: &[C],
    targets: &[[f64; 3]],
) -> (Vec<C>, Vec<[C; 3]>) {
    let zero = C::new(0.0, 0.0);
    let mut phi = vec![zero; targets.len()];
    let mut grad = vec![[zero; 3]; targets.len()];
    let kc = C::new(k.re(), k.im());
    p2p::direct_sum(kc, sources, charges, targets, &mut phi, Some(&mut grad));
    let scale = 1.0 / (4.0 * PI);
    (
        phi.iter().map(|&x| x * scale).collect(),
        grad.iter().map(|g| g.map(|x| x * scale)).collect(),
    )
}

/// Vector v of `output` as complex values (imaginary parts zero on Yukawa's real path);
/// no gradients if the output has none.
fn output_of(output: &Output<f64>, v: usize) -> (Vec<C>, Vec<[C; 3]>) {
    let phi = output.potential_of(v);
    let im = |i: usize| phi.im.map_or(0.0, |im| im[i]);
    let potential = (0..phi.re.len())
        .map(|i| C::new(phi.re[i], im(i)))
        .collect();
    let Some(g) = output.gradient_of(v) else {
        return (potential, Vec::new());
    };
    let gradient = (0..g.re.len())
        .map(|i| core::array::from_fn(|a| C::new(g.re[i][a], g.im.map_or(0.0, |im| im[i][a]))))
        .collect();
    (potential, gradient)
}

#[test]
fn sizes_are_the_table_of_3_15() {
    for kind in KINDS {
        for charges in CHARGES {
            for m in [1, 3] {
                let d = Densities::new(kind, charges, m);
                let c = if charges == ChargeKind::Real { 1 } else { 2 };
                // §3.15, "Storage": (multipole, local) planes of each path.
                let (mp, lp) = match (kind, charges) {
                    (WavenumberKind::Yukawa, ChargeKind::Real) => (1, 1),
                    (WavenumberKind::Real, ChargeKind::Real) => (1, 2),
                    _ => (2, 2),
                };
                for p in [0, 5] {
                    let nc = (p + 1) * (p + 1);
                    assert_eq!(d.multipole_size(p), mp * nc * m, "{kind:?} {charges:?}");
                    assert_eq!(d.local_size(p), lp * nc * m, "{kind:?} {charges:?}");
                    // One complex charge vector costs one complex expansion, never two.
                    assert!(d.multipole_size(p) <= 2 * nc * m);
                }
                assert_eq!(d.columns(), c * m);
                assert_eq!(d.source_point_size(), 3 + c * m);
                assert_eq!(d.output_columns(), lp * m);
                assert_eq!(d.target_output_point_size(false), lp * m);
                assert_eq!(d.target_output_point_size(true), 4 * lp * m);
            }
        }
    }
}

#[test]
fn chunk_indices_are_the_several_vector_layout() {
    let n = 7;
    for kind in KINDS {
        for charges in CHARGES {
            for m in [1, 3] {
                let d = Densities::new(kind, charges, m);
                let (cols, out) = (d.columns(), d.output_columns());
                let c = cols / m;
                // §3.13, "Several charge vectors", with M columns as the vectors: source j's
                // strength w at 3n + j M + w; column o's φ̂ at o n + i, its ĝ at
                // M n + 3 (o n + i).
                for j in 0..n {
                    for v in 0..m {
                        for part in 0..c {
                            let w = c * v + part;
                            assert_eq!(d.charge_index(n, j, v, part), 3 * n + j * cols + w);
                        }
                    }
                }
                for v in 0..m {
                    for part in 0..out / m {
                        let o = d.output_column(v, part);
                        assert_eq!(o, (out / m) * v + part);
                        for i in 0..n {
                            assert_eq!(d.potential_index(n, o, i), o * n + i);
                            for a in 0..3 {
                                let g = d.gradient_index(n, o, i, a);
                                assert_eq!(g, out * n + 3 * (o * n + i) + a);
                            }
                        }
                    }
                }
            }
        }
    }
    // Yukawa's complex vectors are its real path's 2m vectors, index for index.
    let complex = Densities::new(WavenumberKind::Yukawa, ChargeKind::Complex, 3);
    let real = Densities::new(WavenumberKind::Yukawa, ChargeKind::Real, 6);
    assert_eq!(complex.multipole_size(4), real.multipole_size(4));
    assert_eq!(complex.local_size(4), real.local_size(4));
    for j in 0..n {
        for v in 0..3 {
            for part in 0..2 {
                assert_eq!(
                    complex.charge_index(n, j, v, part),
                    real.charge_index(n, j, 2 * v + part, 0)
                );
                assert_eq!(
                    complex.output_column(v, part),
                    real.output_column(2 * v + part, 0)
                );
            }
        }
    }
}

#[test]
fn charges_in() {
    let d = Densities::new(WavenumberKind::Real, ChargeKind::Complex, 2);
    let (re0, im0) = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let (re1, im1) = ([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let vectors = [
        Charges::Complex {
            re: &re0[..],
            im: &im0[..],
        },
        Charges::Complex {
            re: &re1[..],
            im: &im1[..],
        },
    ];
    assert_eq!(d.check_charges(&vectors, 3), Ok(()));
    // The leaf holds sources 2 and 0, in that order.
    let mut strengths = [0.0; 8];
    d.write_leaf_charges(&vectors, &[2, 0], &mut strengths);
    assert_eq!(strengths, [3.0, 6.0, 9.0, 12.0, 1.0, 4.0, 7.0, 10.0]);
    for (j, source) in [(0, 2), (1, 0)] {
        for (v, [re, im]) in [[re0, im0], [re1, im1]].iter().enumerate() {
            assert_eq!(strengths[d.charge_index(2, j, v, 0) - 6], re[source]);
            assert_eq!(strengths[d.charge_index(2, j, v, 1) - 6], im[source]);
        }
    }

    assert_eq!(
        d.check_charges(&vectors[..1], 3),
        Err(DensityError::VectorCount {
            expected: 2,
            got: 1
        })
    );
    let real = [Charges::Real(&re0[..]), vectors[1]];
    assert_eq!(
        d.check_charges(&real, 3),
        Err(DensityError::ChargeKind { vector: 0 })
    );
    let short = [
        vectors[0],
        Charges::Complex {
            re: &re1[..],
            im: &im1[..2],
        },
    ];
    assert_eq!(
        d.check_charges(&short, 3),
        Err(DensityError::Length {
            vector: 1,
            expected: 3,
            got: 2
        })
    );

    // Real charges: one column per vector, and a short plane.
    let d = Densities::new(WavenumberKind::Complex, ChargeKind::Real, 2);
    let vectors = [Charges::Real(&re0[..]), Charges::Real(&re1[..])];
    assert_eq!(d.check_charges(&vectors, 3), Ok(()));
    let mut strengths = [0.0; 4];
    d.write_leaf_charges(&vectors, &[1, 2], &mut strengths);
    assert_eq!(strengths, [2.0, 8.0, 3.0, 9.0]);
    let short = [Charges::Real(&re0[..]), Charges::Real(&re1[..1])];
    assert_eq!(
        d.check_charges(&short, 3),
        Err(DensityError::Length {
            vector: 1,
            expected: 3,
            got: 1
        })
    );
}

#[test]
fn p2m_is_the_reference_bit_for_bit() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let (n, m, p) = (23, 3, 9);
    let mut rng = Rng(11);
    let points = leaf_points(&domain, key, n, &mut rng);
    let mut scratch = LeafScratch::new(12);
    let mut ws = Workspace::new(p);
    let len = (p + 1) * (p + 1);
    for kind in KINDS {
        let k = wavenumber(kind, 21.0);
        let (level, frame) = leaf_level(&k, r);
        for charges in CHARGES {
            let d = Densities::new(kind, charges, m);
            let planes = charge_planes(charges, m, n, &mut rng);
            let chunk = source_chunk(&d, &domain, key, &points, &planes);
            let u = chunk_points(&chunk, n);
            let mut multipoles = vec![0.0; d.multipole_size(p)];
            d.p2m(&k, r, p, &chunk, n, &mut scratch, &mut multipoles);
            let size = d.multipole_size(p) / m;
            for v in 0..m {
                let got = &multipoles[v * size..(v + 1) * size];
                let mut want = vec![0.0; size];
                match (kind, charges) {
                    (WavenumberKind::Yukawa, _) => {
                        // Re q's real expansion, then Im q's for a complex vector.
                        for (part, plane) in want.chunks_exact_mut(len).enumerate() {
                            let q = &planes[v][part];
                            leaf::p2m_yukawa(p, &level, &frame, &u, q, &mut ws, plane);
                        }
                    }
                    (WavenumberKind::Real, ChargeKind::Real) => {
                        let mut full = vec![0.0; 2 * len];
                        let q = complex_of(&planes, v);
                        leaf::p2m(p, &level, &frame, &u, &q, &mut ws, &mut full);
                        assert!(full[len..].iter().all(|&x| x == 0.0), "Im plane of real k");
                        want.copy_from_slice(&full[..len]);
                    }
                    _ => {
                        let q = complex_of(&planes, v);
                        leaf::p2m(p, &level, &frame, &u, &q, &mut ws, &mut want);
                    }
                }
                assert_eq!(got, &want[..], "{kind:?} {charges:?} vector {v}");
                // Vector v is the one-vector P2M of vector v.
                let one = Densities::new(kind, charges, 1);
                let one_chunk = source_chunk(&one, &domain, key, &points, &planes[v..=v]);
                let mut single = vec![0.0; one.multipole_size(p)];
                one.p2m(&k, r, p, &one_chunk, n, &mut scratch, &mut single);
                assert_eq!(got, &single[..], "{kind:?} {charges:?} vector {v} alone");
            }
        }
    }
}

/// P2M and L2P add to what their outputs hold: into a non-zero buffer, the buffer plus the
/// operator's result into zeros, to rounding (bit for bit for the real multipole of real
/// k, which is summed in scratch and then added).
#[test]
fn operators_accumulate() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let (n, m, p) = (13, 2, 7);
    let mut rng = Rng(31);
    let points = leaf_points(&domain, key, n, &mut rng);
    let mut scratch = LeafScratch::new(p);
    let close = |a: &[f64], b: &[f64], what: &str| {
        let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let norm: f64 = b.iter().map(|y| y * y).sum();
        assert!((diff / norm).sqrt() < 1e-14, "{what}");
    };
    for kind in KINDS {
        let k = wavenumber(kind, 21.0);
        for charges in CHARGES {
            let d = Densities::new(kind, charges, m);
            let what = format!("{kind:?} {charges:?}");
            let planes = charge_planes(charges, m, n, &mut rng);
            let chunk = source_chunk(&d, &domain, key, &points, &planes);
            let start: Vec<f64> = (0..d.multipole_size(p)).map(|_| rng.next()).collect();
            let mut fresh = vec![0.0; start.len()];
            d.p2m(&k, r, p, &chunk, n, &mut scratch, &mut fresh);
            let mut added = start.clone();
            d.p2m(&k, r, p, &chunk, n, &mut scratch, &mut added);
            let sum: Vec<f64> = start.iter().zip(&fresh).map(|(a, b)| a + b).collect();
            if (kind, charges) == (WavenumberKind::Real, ChargeKind::Real) {
                assert_eq!(added, sum, "{what}: P2M");
            } else {
                close(&added, &sum, &format!("{what}: P2M"));
            }

            let locals: Vec<f64> = (0..d.local_size(p)).map(|_| rng.next()).collect();
            let u = &chunk[..3 * n];
            let len = n * d.target_output_point_size(true);
            let start: Vec<f64> = (0..len).map(|_| rng.next()).collect();
            let mut fresh = vec![0.0; len];
            d.l2p(&k, r, p, &locals, u, &mut scratch, &mut fresh, true);
            let mut added = start.clone();
            d.l2p(&k, r, p, &locals, u, &mut scratch, &mut added, true);
            let sum: Vec<f64> = start.iter().zip(&fresh).map(|(a, b)| a + b).collect();
            assert_eq!(added, sum, "{what}: L2P");
        }
    }
}

#[test]
fn yukawa_complex_vectors_are_two_real_vectors() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let (n, m, p) = (17, 2, 8);
    let mut rng = Rng(5);
    let points = leaf_points(&domain, key, n, &mut rng);
    let k = wavenumber(WavenumberKind::Yukawa, 30.0);
    let complex = Densities::new(WavenumberKind::Yukawa, ChargeKind::Complex, m);
    let real = Densities::new(WavenumberKind::Yukawa, ChargeKind::Real, 2 * m);
    let planes = charge_planes(ChargeKind::Complex, m, n, &mut rng);
    // The 2m real vectors Re q⁰, Im q⁰, Re q¹, Im q¹.
    let split: Vec<[Vec<f64>; 2]> = planes
        .iter()
        .flat_map(|[re, im]| [[re.clone(), vec![0.0; n]], [im.clone(), vec![0.0; n]]])
        .collect();
    let chunk = source_chunk(&complex, &domain, key, &points, &planes);
    let real_chunk = source_chunk(&real, &domain, key, &points, &split);
    assert_eq!(chunk, real_chunk);

    let mut scratch = LeafScratch::new(p);
    let (mut a, mut b) = (
        vec![0.0; complex.multipole_size(p)],
        vec![0.0; real.multipole_size(p)],
    );
    complex.p2m(&k, r, p, &chunk, n, &mut scratch, &mut a);
    real.p2m(&k, r, p, &real_chunk, n, &mut scratch, &mut b);
    assert_eq!(a, b);

    let locals: Vec<f64> = (0..complex.local_size(p)).map(|_| rng.next()).collect();
    let u = &chunk[..3 * n];
    for gradients in [false, true] {
        let size = n * complex.target_output_point_size(gradients);
        assert_eq!(size, n * real.target_output_point_size(gradients));
        let (mut a, mut b) = (vec![0.0; size], vec![0.0; size]);
        complex.l2p(&k, r, p, &locals, u, &mut scratch, &mut a, gradients);
        real.l2p(&k, r, p, &locals, u, &mut scratch, &mut b, gradients);
        assert_eq!(a, b);
    }
}

/// A reference L2P's output column, φ̂, ĝ, and φ̂ of the call without gradients.
type Expected = (usize, Vec<f64>, Vec<[f64; 3]>, Vec<f64>);

#[test]
fn l2p_is_the_reference_bit_for_bit() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let (n, m, p) = (19, 3, 9);
    let mut rng = Rng(23);
    let points = leaf_points(&domain, key, n, &mut rng);
    let u: Vec<[f64; 3]> = points
        .iter()
        .map(|&x| geometry::leaf_coordinates(x, key, &domain))
        .collect();
    let flat: Vec<f64> = u.iter().flatten().copied().collect();
    let mut scratch = LeafScratch::new(p);
    let mut ws = Workspace::new(p);
    let len = (p + 1) * (p + 1);
    let zero = C::new(0.0, 0.0);
    for kind in KINDS {
        let k = wavenumber(kind, 21.0);
        let (level, frame) = leaf_level(&k, r);
        for charges in CHARGES {
            let d = Densities::new(kind, charges, m);
            let locals: Vec<f64> = (0..d.local_size(p)).map(|_| rng.next()).collect();
            let mut output = vec![0.0; n * d.target_output_point_size(true)];
            d.l2p(&k, r, p, &locals, &flat, &mut scratch, &mut output, true);
            let mut plain = vec![0.0; n * d.target_output_point_size(false)];
            d.l2p(&k, r, p, &locals, &flat, &mut scratch, &mut plain, false);
            let size = d.local_size(p) / m;
            for v in 0..m {
                let local = &locals[v * size..(v + 1) * size];
                // (column, φ̂, ĝ, φ̂ without gradients) of the reference.
                let mut want: Vec<Expected> = Vec::new();
                if kind == WavenumberKind::Yukawa {
                    for (part, plane) in local.chunks_exact(len).enumerate() {
                        let mut alone = vec![0.0; n];
                        leaf::l2p_yukawa(p, &level, &frame, plane, &u, &mut ws, &mut alone, None);
                        let (mut phi, mut g) = (vec![0.0; n], vec![[0.0; 3]; n]);
                        leaf::l2p_yukawa(
                            p,
                            &level,
                            &frame,
                            plane,
                            &u,
                            &mut ws,
                            &mut phi,
                            Some(&mut g),
                        );
                        want.push((d.output_column(v, part), phi, g, alone));
                    }
                } else {
                    let mut alone = vec![zero; n];
                    leaf::l2p(p, &level, &frame, local, &u, &mut ws, &mut alone, None);
                    let (mut phi, mut g) = (vec![zero; n], vec![[zero; 3]; n]);
                    leaf::l2p(
                        p,
                        &level,
                        &frame,
                        local,
                        &u,
                        &mut ws,
                        &mut phi,
                        Some(&mut g),
                    );
                    for part in 0..2 {
                        let pick = |c: C| if part == 0 { c.re } else { c.im };
                        want.push((
                            d.output_column(v, part),
                            phi.iter().map(|&c| pick(c)).collect(),
                            g.iter().map(|gi| gi.map(pick)).collect(),
                            alone.iter().map(|&c| pick(c)).collect(),
                        ));
                    }
                }
                for (column, phi, g, alone) in want {
                    for i in 0..n {
                        assert_eq!(output[d.potential_index(n, column, i)], phi[i]);
                        assert_eq!(plain[d.potential_index(n, column, i)], alone[i]);
                        for a in 0..3 {
                            assert_eq!(output[d.gradient_index(n, column, i, a)], g[i][a]);
                        }
                    }
                }
            }
        }
    }
}

/// P2M from the leaf's chunk, M2P by the reference at far targets, the output pass:
/// against the direct sum.
#[test]
fn p2m_end_to_end_against_the_direct_sum() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let (n, nt, m) = (40, 30, 2);
    let mut rng = Rng(7);
    let points = leaf_points(&domain, key, n, &mut rng);
    let targets = far_points(&domain, key, nt, 4.0, &mut rng);
    let target_u: Vec<[f64; 3]> = targets
        .iter()
        .map(|x| {
            let c = geometry::centre(key, &domain);
            core::array::from_fn(|a| (x[a] - c[a]) / r)
        })
        .collect();
    let mut worst = 0.0f64;
    let zero = C::new(0.0, 0.0);
    // (|k|, p): k̂ = |k| r of 0.875, 5 and 10 (|k| D = 14, 80 and 160).
    for (size, p) in [(7.0, 20), (40.0, 30), (80.0, 40)] {
        let mut scratch = LeafScratch::new(p);
        let mut ws = Workspace::new(p);
        let len = (p + 1) * (p + 1);
        for kind in KINDS {
            let k = wavenumber(kind, size);
            let (level, frame) = leaf_level(&k, r);
            for charges in CHARGES {
                let d = Densities::new(kind, charges, m);
                let planes = charge_planes(charges, m, n, &mut rng);
                let chunk = source_chunk(&d, &domain, key, &points, &planes);
                let mut multipoles = vec![0.0; d.multipole_size(p)];
                d.p2m(&k, r, p, &chunk, n, &mut scratch, &mut multipoles);

                // M2P into the target output chunk of a leaf of the same level, then its
                // output pass.
                let mut out_chunk = vec![0.0; nt * d.target_output_point_size(true)];
                let size_v = d.multipole_size(p) / m;
                for v in 0..m {
                    let multipole = &multipoles[v * size_v..(v + 1) * size_v];
                    let mut add = |column: usize, phi: &[f64], g: &[[f64; 3]]| {
                        for i in 0..nt {
                            out_chunk[d.potential_index(nt, column, i)] += phi[i];
                            for a in 0..3 {
                                out_chunk[d.gradient_index(nt, column, i, a)] += g[i][a];
                            }
                        }
                    };
                    if kind == WavenumberKind::Yukawa {
                        for (part, plane) in multipole.chunks_exact(len).enumerate() {
                            let (mut phi, mut g) = (vec![0.0; nt], vec![[0.0; 3]; nt]);
                            leaf::m2p_yukawa(
                                p,
                                &level,
                                &frame,
                                plane,
                                &target_u,
                                &mut ws,
                                &mut phi,
                                Some(&mut g),
                            );
                            add(d.output_column(v, part), &phi, &g);
                        }
                    } else {
                        let mut full = vec![0.0; 2 * len];
                        full[..multipole.len()].copy_from_slice(multipole);
                        let (mut phi, mut g) = (vec![zero; nt], vec![[zero; 3]; nt]);
                        leaf::m2p(
                            p,
                            &level,
                            &frame,
                            &full,
                            &target_u,
                            &mut ws,
                            &mut phi,
                            Some(&mut g),
                        );
                        for part in 0..2 {
                            let pick = |c: C| if part == 0 { c.re } else { c.im };
                            let phi: Vec<f64> = phi.iter().map(|&c| pick(c)).collect();
                            let g: Vec<[f64; 3]> = g.iter().map(|gi| gi.map(pick)).collect();
                            add(d.output_column(v, part), &phi, &g);
                        }
                    }
                }
                let mut output = Output::new(&d, nt, true);
                let order: Vec<usize> = (0..nt).collect();
                d.write_leaf_output(&out_chunk, r, true, &order, &mut output);

                for v in 0..m {
                    let (phi, g) = output_of(&output, v);
                    let (want_phi, want_g) = exact(&k, &points, &complex_of(&planes, v), &targets);
                    let flat = |g: &[[C; 3]]| g.iter().flatten().copied().collect::<Vec<C>>();
                    let e_phi = relative_l2(&phi, &want_phi);
                    let e_g = relative_l2(&flat(&g), &flat(&want_g));
                    let what = format!("{kind:?} {charges:?} |k| {size} p {p} vector {v}");
                    assert!(e_phi < 1e-10, "{what}: φ {e_phi:.2e}");
                    worst = worst.max(e_phi).max(e_g);
                    assert!(e_g < 1e-10, "{what}: ∇φ {e_g:.2e}");
                    assert_eq!(phi.len(), nt);
                    assert_eq!(
                        output.potential_of(v).im.is_some(),
                        d.output_columns() == 2 * m
                    );
                }
            }
        }
    }
    println!("p2m_end_to_end_against_the_direct_sum: worst relative L2 {worst:.2e}");
}

/// Far sources, P2L by the reference into the leaf, L2P into its target output chunk, the
/// output pass in a permuted target order: against the direct sum, the planes at their
/// places of §3.15.
#[test]
fn l2p_end_to_end_against_the_direct_sum() {
    let (domain, key) = leaf_box();
    let r = geometry::radius(3, &domain);
    let c = geometry::centre(key, &domain);
    let (ns, nt, m) = (30, 40, 2);
    let mut rng = Rng(13);
    let sources = far_points(&domain, key, ns, 4.0, &mut rng);
    let source_u: Vec<[f64; 3]> = sources
        .iter()
        .map(|x| core::array::from_fn(|a| (x[a] - c[a]) / r))
        .collect();
    let targets = leaf_points(&domain, key, nt, &mut rng);
    let flat_u: Vec<f64> = targets
        .iter()
        .flat_map(|&x| geometry::leaf_coordinates::<f64>(x, key, &domain))
        .collect();
    // The leaf's point i is the caller's target nt − 1 − i.
    let order: Vec<usize> = (0..nt).rev().collect();
    let caller_targets: Vec<[f64; 3]> = (0..nt).map(|t| targets[nt - 1 - t]).collect();
    let mut worst = 0.0f64;
    for (size, p) in [(7.0, 20), (40.0, 30), (80.0, 40)] {
        let mut scratch = LeafScratch::new(p);
        let mut ws = Workspace::new(p);
        let len = (p + 1) * (p + 1);
        for kind in KINDS {
            let k = wavenumber(kind, size);
            let (level, frame) = leaf_level(&k, r);
            for charges in CHARGES {
                let d = Densities::new(kind, charges, m);
                let planes = charge_planes(charges, m, ns, &mut rng);
                let size_v = d.local_size(p) / m;
                let mut locals = vec![0.0; d.local_size(p)];
                for (v, local) in locals.chunks_exact_mut(size_v).enumerate() {
                    if kind == WavenumberKind::Yukawa {
                        for (part, plane) in local.chunks_exact_mut(len).enumerate() {
                            let q = &planes[v][part];
                            leaf::p2l_yukawa(p, &level, &frame, &source_u, q, &mut ws, plane);
                        }
                    } else {
                        let q = complex_of(&planes, v);
                        leaf::p2l(p, &level, &frame, &source_u, &q, &mut ws, local);
                    }
                }
                let mut out_chunk = vec![0.0; nt * d.target_output_point_size(true)];
                d.l2p(
                    &k,
                    r,
                    p,
                    &locals,
                    &flat_u,
                    &mut scratch,
                    &mut out_chunk,
                    true,
                );
                let mut output = Output::new(&d, nt, true);
                d.write_leaf_output(&out_chunk, r, true, &order, &mut output);
                // Without gradients: L2P and the output pass of potentials only.
                let mut plain_chunk = vec![0.0; nt * d.target_output_point_size(false)];
                d.l2p(
                    &k,
                    r,
                    p,
                    &locals,
                    &flat_u,
                    &mut scratch,
                    &mut plain_chunk,
                    false,
                );
                let mut plain = Output::new(&d, nt, false);
                d.write_leaf_output(&plain_chunk, r, false, &order, &mut plain);
                assert!(plain.gradient.is_none() && plain.gradient_of(0).is_none());

                assert_eq!(output.is_complex(), d.output_columns() == 2 * m);
                assert_eq!(output.ntargets(), nt);
                for v in 0..m {
                    let (phi, g) = output_of(&output, v);
                    let q = complex_of(&planes, v);
                    let (want_phi, want_g) = exact(&k, &sources, &q, &caller_targets);
                    let flat = |g: &[[C; 3]]| g.iter().flatten().copied().collect::<Vec<C>>();
                    let e_phi = relative_l2(&phi, &want_phi);
                    let e_g = relative_l2(&flat(&g), &flat(&want_g));
                    let (plain_phi, _) = output_of(&plain, v);
                    let e_plain = relative_l2(&plain_phi, &want_phi);
                    let what = format!("{kind:?} {charges:?} |k| {size} p {p} vector {v}");
                    assert!(e_phi < 1e-10, "{what}: φ {e_phi:.2e}");
                    worst = worst.max(e_phi).max(e_g).max(e_plain);
                    assert!(e_g < 1e-10, "{what}: ∇φ {e_g:.2e}");
                    assert!(e_plain < 1e-10, "{what}: φ without ∇φ {e_plain:.2e}");
                    // §3.15's order: vector v's Re φ at column 2v, Im φ at 2v + 1, and its
                    // Re ∇φ triples at column 2v, Im ∇φ at 2v + 1 (v alone on Yukawa's real
                    // path).
                    let gradient = output.gradient.as_deref().unwrap();
                    let planes = if output.is_complex() { 2 } else { 1 };
                    let column = |o: usize| o * nt..(o + 1) * nt;
                    for out in [&output, &plain] {
                        let phi = out.potential_of(v);
                        assert_eq!(phi.re, &out.potential[column(planes * v)]);
                        let im = output
                            .is_complex()
                            .then(|| &out.potential[column(2 * v + 1)]);
                        assert_eq!(phi.im, im);
                    }
                    let g = output.gradient_of(v).unwrap();
                    assert_eq!(g.re, &gradient[column(planes * v)]);
                    let im = output.is_complex().then(|| &gradient[column(2 * v + 1)]);
                    assert_eq!(g.im, im);
                }
            }
        }
    }
    println!("l2p_end_to_end_against_the_direct_sum: worst relative L2 {worst:.2e}");
}
