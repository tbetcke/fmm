//! The Helmholtz FMM beside FMM3D's `hfmm3d` (Helmholtz CH3.6;
//! docs/design/helmholtz-fmm-proposal.md §9.2, the H3 row): the files of
//! `tools/fmm3d/driver.f90`'s `helmholtz` mode, FMM3D's errors at the sampled targets of a
//! [`helmholtz_accuracy`](crate::helmholtz_accuracy) case, and the full-target crosscheck.
//!
//! - **Files.** [`write_problem`] writes a case's points, one charge vector and the
//!   sampled targets as the driver's problem file (`NDFMMHP1`); [`read_sampled`] reads its
//!   output at the sampled sources (`NDFMMHO1`) and [`read_full`] its output at every source
//!   (`NDFMMHF1`, the driver's FULL = 1); [`seconds`] reads a call's time from its
//!   `helmholtz:` line. The byte layouts are those of the driver's header: little-endian,
//!   complex values as (re, im) pairs of f64, ∇φ by point and then axis.
//! - **Sampled errors.** [`sample_errors`]: §9.1's relative ℓ2 norms of φ and ∇φ of FMM3D's
//!   values against the [`Oracle`] that measures our FMM, so that both codes are measured
//!   against the same direct sum at the same targets.
//! - **Crosscheck.** [`crosscheck`] (our distributed output) and [`crosscheck_values`]
//!   (FMM3D's output) against a reference at every point, normally FMM3D at a tight eps:
//!   the relative ℓ2 errors of φ and ∇φ over all N targets, and per box of a level of the
//!   FMM's domain ([`Grid`]) the largest error at a target, |φ − φ_ref| over the RMS of
//!   φ_ref over all targets (‖∇φ − ∇φ_ref‖ over the RMS of ‖∇φ_ref‖ for the gradient). A
//!   box's error is then comparable to the ℓ2 errors, and an error confined to a few boxes,
//!   which 1,000 sampled targets can miss, shows as a box far above the ℓ2 error.

use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::Path;

use mpi::datatype::PartitionMut;
use mpi::traits::CommunicatorCollectives;
use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::Output;
use nd_fmm_math::Complex;

use crate::helmholtz_accuracy::{Oracle, Problem};

/// The magic of the driver's problem file.
pub const PROBLEM_MAGIC: &[u8; 8] = b"NDFMMHP1";
/// The magic of the driver's output at the sampled sources (and targets).
pub const SAMPLED_MAGIC: &[u8; 8] = b"NDFMMHO1";
/// The magic of the driver's output at every source (FULL = 1).
pub const FULL_MAGIC: &[u8; 8] = b"NDFMMHF1";

/// Complex potentials and gradients at a list of targets, in f64.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    /// φ per target.
    pub potential: Vec<Complex<f64>>,
    /// ∇φ per target.
    pub gradient: Vec<[Complex<f64>; 3]>,
}

impl Values {
    /// The number of targets.
    pub fn len(&self) -> usize {
        self.potential.len()
    }

    /// Whether there are no targets.
    pub fn is_empty(&self) -> bool {
        self.potential.is_empty()
    }

    /// The values at the positions `positions`, in their order.
    pub fn at(&self, positions: &[usize]) -> Values {
        Values {
            potential: positions.iter().map(|&i| self.potential[i]).collect(),
            gradient: positions.iter().map(|&i| self.gradient[i]).collect(),
        }
    }
}

/// The driver's problem file of `problem` with charge vector `vector` (Im q = 0 for real
/// charges), the sampled targets as 0-based source indices and no extra targets; `seed`
/// is recorded in the header.
pub fn problem_bytes(problem: &Problem, vector: usize, seed: u64) -> Vec<u8> {
    let n = problem.len();
    let s = problem.sample.len();
    let mut out = Vec::with_capacity(40 + 40 * n + 8 * s);
    out.extend_from_slice(PROBLEM_MAGIC);
    for v in [n as i64, s as i64, 0, seed as i64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for x in problem.points.iter().flatten() {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for q in problem.complex_charges(vector) {
        out.extend_from_slice(&q.re.to_le_bytes());
        out.extend_from_slice(&q.im.to_le_bytes());
    }
    for &i in &problem.sample {
        out.extend_from_slice(&(i as i64).to_le_bytes());
    }
    out
}

/// Writes [`problem_bytes`] to `path`.
///
/// # Errors
///
/// The file system's.
pub fn write_problem(path: &Path, problem: &Problem, vector: usize, seed: u64) -> io::Result<()> {
    std::fs::write(path, problem_bytes(problem, vector, seed))
}

/// An `InvalidData` error naming `path`.
fn invalid(path: &Path, message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {message}", path.display()),
    )
}

/// The little-endian f64 values of `bytes`.
fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| f64::from_le_bytes(c))
        .collect()
}

/// The little-endian i64 at `offset`.
fn integer(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("eight bytes"))
}

/// `count` potentials and then `count` gradients from the f64 values `v`.
fn values(v: &[f64], count: usize) -> Values {
    let c = |j: usize| Complex::new(v[2 * j], v[2 * j + 1]);
    Values {
        potential: (0..count).map(c).collect(),
        gradient: (0..count)
            .map(|i| std::array::from_fn(|a| c(count + 3 * i + a)))
            .collect(),
    }
}

/// The f64 values of `values` in the driver's order: every φ, then ∇φ by point and axis,
/// complex values as (re, im).
fn value_bytes(values: &Values) -> impl Iterator<Item = u8> + '_ {
    let gradients = values.gradient.iter().flatten();
    values
        .potential
        .iter()
        .chain(gradients)
        .flat_map(|v| [v.re.to_le_bytes(), v.im.to_le_bytes()])
        .flatten()
}

/// `values` at the sampled sources in the byte layout of the driver's `NDFMMHO1` output (no
/// extra targets), as [`parse_sampled`] reads it; for tests without FMM3D.
pub fn sampled_bytes(values: &Values) -> Vec<u8> {
    let mut out = SAMPLED_MAGIC.to_vec();
    out.extend_from_slice(&(values.len() as i64).to_le_bytes());
    out.extend_from_slice(&0i64.to_le_bytes());
    out.extend(value_bytes(values));
    out
}

/// `values` at every source in the byte layout of the driver's `NDFMMHF1` output, as
/// [`parse_full`] reads it; for tests without FMM3D.
pub fn full_bytes(values: &Values) -> Vec<u8> {
    let mut out = FULL_MAGIC.to_vec();
    out.extend_from_slice(&(values.len() as i64).to_le_bytes());
    out.extend(value_bytes(values));
    out
}

/// Parses the driver's output at the sampled sources (`NDFMMHO1`): φ and ∇φ at the
/// sampled sources, in the problem's order of the samples; values at extra targets, if
/// any, are ignored.
///
/// # Errors
///
/// `InvalidData` (naming `path`) for another magic or a length that does not match.
pub fn parse_sampled(path: &Path, bytes: &[u8]) -> io::Result<Values> {
    if bytes.len() < 24 || &bytes[..8] != SAMPLED_MAGIC {
        return Err(invalid(path, "not an NDFMMHO1 file"));
    }
    let (s, t) = (integer(bytes, 8), integer(bytes, 16));
    if s < 0 || t < 0 || bytes.len() != 24 + 64 * (s + t) as usize {
        return Err(invalid(path, "its length does not match its header"));
    }
    let s = s as usize;
    Ok(values(&floats(&bytes[24..24 + 64 * s]), s))
}

/// Parses the driver's output at every source (`NDFMMHF1`).
///
/// # Errors
///
/// `InvalidData` (naming `path`) for another magic or a length that does not match.
pub fn parse_full(path: &Path, bytes: &[u8]) -> io::Result<Values> {
    if bytes.len() < 16 || &bytes[..8] != FULL_MAGIC {
        return Err(invalid(path, "not an NDFMMHF1 file"));
    }
    let n = integer(bytes, 8);
    if n < 0 || bytes.len() != 16 + 64 * n as usize {
        return Err(invalid(path, "its length does not match its header"));
    }
    Ok(values(&floats(&bytes[16..]), n as usize))
}

/// Reads [`parse_sampled`] from `path`.
///
/// # Errors
///
/// The file system's, and [`parse_sampled`]'s.
pub fn read_sampled(path: &Path) -> io::Result<Values> {
    parse_sampled(path, &std::fs::read(path)?)
}

/// Reads [`parse_full`] from `path`.
///
/// # Errors
///
/// The file system's, and [`parse_full`]'s.
pub fn read_full(path: &Path) -> io::Result<Values> {
    parse_full(path, &std::fs::read(path)?)
}

/// From the driver's output `log`, the seconds and OpenMP threads of the `helmholtz:` line
/// of `token` (the last such line), if any.
pub fn seconds(log: &str, token: &str) -> Option<(f64, usize)> {
    let mut found = None;
    for line in log.lines() {
        let Some(rest) = line.strip_prefix("helmholtz: token=") else {
            continue;
        };
        let words: Vec<&str> = rest.split_whitespace().collect();
        if words.first() != Some(&token) {
            continue;
        }
        // Fortran pads a number after `key=` with blanks, so the value may be the next word.
        let field = |key: &str| {
            let at = words.iter().position(|w| w.starts_with(key))?;
            let value = &words[at][key.len()..];
            Some(
                if value.is_empty() {
                    *words.get(at + 1)?
                } else {
                    value
                }
                .to_owned(),
            )
        };
        let seconds = field("seconds=").and_then(|v| v.replace(['D', 'd'], "E").parse().ok());
        let threads = field("threads=").and_then(|v| v.parse().ok());
        if let (Some(s), Some(t)) = (seconds, threads) {
            found = Some((s, t));
        }
    }
    found
}

/// §9.1's relative ℓ2 errors [φ, ∇φ] of `values` at the sampled targets against vector
/// `vector` of `oracle` (the same targets, in the order of the samples).
///
/// # Panics
///
/// If `values` and the oracle hold different numbers of targets.
pub fn sample_errors(values: &Values, oracle: &Oracle, vector: usize) -> [f64; 2] {
    let (phi, grad) = (&oracle.potential[vector], &oracle.gradient[vector]);
    assert_eq!(values.len(), phi.len(), "one value per sampled target");
    let mut sums = [0.0f64; 4];
    for i in 0..phi.len() {
        sums[0] += (values.potential[i] - phi[i]).norm_sqr();
        sums[1] += phi[i].norm_sqr();
        if !grad.is_empty() {
            for (v, g) in values.gradient[i].iter().zip(&grad[i]) {
                sums[2] += (v - g).norm_sqr();
                sums[3] += g.norm_sqr();
            }
        }
    }
    let ratio = |e: f64, r: f64| if r > 0.0 { (e / r).sqrt() } else { 0.0 };
    [ratio(sums[0], sums[1]), ratio(sums[2], sums[3])]
}

/// The boxes of one level of a domain, the crosscheck's unit of place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// The domain (the FMM's, so that the boxes are its tree's).
    pub domain: Domain,
    /// The level, at most 21.
    pub level: usize,
}

impl Grid {
    /// The box that holds `x`, as (i, j, k) packed into 21 bits each (i lowest); a point
    /// outside the domain is in the nearest box, a point on its upper faces in the last.
    ///
    /// # Panics
    ///
    /// If the level exceeds 21.
    pub fn box_of(&self, x: [f64; 3]) -> u64 {
        assert!(self.level <= 21, "at most 21 levels pack into a u64");
        let (lower, side) = (self.domain.lower(), self.domain.side());
        let cells = 1u64 << self.level;
        (0..3).fold(0u64, |key, a| {
            let t = ((x[a] - lower[a]) / side * cells as f64).floor();
            let i = (t.max(0.0) as u64).min(cells - 1);
            key | (i << (21 * a))
        })
    }
}

/// The full-target crosscheck of one output against a reference (module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Crosscheck {
    /// The relative ℓ2 errors [φ, ∇φ] over every target.
    pub errors: [f64; 2],
    /// The level of the boxes.
    pub level: usize,
    /// Per box that holds a target, the largest error [φ, ∇φ] at one of its targets,
    /// relative to the RMS of the reference over all targets; sorted by box.
    pub boxes: Vec<(u64, [f64; 2])>,
}

impl Crosscheck {
    /// The largest box error [φ, ∇φ].
    pub fn worst(&self) -> [f64; 2] {
        self.boxes
            .iter()
            .fold([0.0, 0.0], |w, (_, e)| [w[0].max(e[0]), w[1].max(e[1])])
    }
}

/// The local part of a crosscheck: the sums of squares [φ error, ∇φ error] and the box
/// maxima of the absolute errors, over the targets `targets` (global positions) with
/// values from `value`.
fn local_part(
    targets: Range<usize>,
    value: impl Fn(usize) -> (Complex<f64>, [Complex<f64>; 3]),
    reference: &Values,
    points: &[[f64; 3]],
    grid: &Grid,
) -> ([f64; 2], HashMap<u64, [f64; 2]>) {
    let mut sums = [0.0f64; 2];
    let mut boxes: HashMap<u64, [f64; 2]> = HashMap::new();
    for i in targets {
        let (phi, grad) = value(i);
        let dphi = (phi - reference.potential[i]).norm_sqr();
        let dgrad: f64 = (0..3)
            .map(|a| (grad[a] - reference.gradient[i][a]).norm_sqr())
            .sum();
        sums[0] += dphi;
        sums[1] += dgrad;
        let entry = boxes.entry(grid.box_of(points[i])).or_insert([0.0, 0.0]);
        entry[0] = entry[0].max(dphi.sqrt());
        entry[1] = entry[1].max(dgrad.sqrt());
    }
    (sums, boxes)
}

/// The [`Crosscheck`] from the summed squares of the errors and the merged box maxima of
/// the absolute errors, normalised by the reference over every target.
fn finish(
    sums: [f64; 2],
    boxes: HashMap<u64, [f64; 2]>,
    reference: &Values,
    level: usize,
) -> Crosscheck {
    let rphi: f64 = reference.potential.iter().map(|v| v.norm_sqr()).sum();
    let rgrad: f64 = reference
        .gradient
        .iter()
        .flat_map(|g| g.iter().map(|v| v.norm_sqr()))
        .sum();
    let ratio = |e: f64, r: f64| if r > 0.0 { (e / r).sqrt() } else { 0.0 };
    let n = reference.len().max(1) as f64;
    let rms = [(rphi / n).sqrt(), (rgrad / n).sqrt()];
    let scale = |e: f64, r: f64| if r > 0.0 { e / r } else { 0.0 };
    let mut boxes: Vec<(u64, [f64; 2])> = boxes
        .into_iter()
        .map(|(k, e)| (k, [scale(e[0], rms[0]), scale(e[1], rms[1])]))
        .collect();
    boxes.sort_by_key(|&(k, _)| k);
    Crosscheck {
        errors: [ratio(sums[0], rphi), ratio(sums[1], rgrad)],
        level,
        boxes,
    }
}

/// [`Crosscheck`] of FMM3D's (or any) `values` at every point of `points` against
/// `reference` at the same points, with the boxes of `grid`. No communication.
///
/// # Panics
///
/// If `values`, `reference` and `points` differ in length.
pub fn crosscheck_values(
    values: &Values,
    reference: &Values,
    points: &[[f64; 3]],
    grid: &Grid,
) -> Crosscheck {
    assert_eq!(values.len(), points.len(), "one value per point");
    assert_eq!(
        reference.len(),
        points.len(),
        "one reference value per point"
    );
    let (sums, boxes) = local_part(
        0..points.len(),
        |i| (values.potential[i], values.gradient[i]),
        reference,
        points,
        grid,
    );
    finish(sums, boxes, reference, grid.level)
}

/// [`Crosscheck`] of vector `vector` of our FMM's `output`, this rank's targets the
/// contiguous block `block` of `problem`'s points (as
/// [`helmholtz_accuracy::run`](crate::helmholtz_accuracy::run) passes them), against
/// `reference` at every point, with the boxes of `grid`. Without gradients in `output`, ∇φ
/// counts as zero.
///
/// # Collective operation
///
/// On `comm`, every rank with the same reference and grid: the sums are
/// all-reduced and every rank's box maxima all-gathered, so every rank returns the same.
///
/// # Panics
///
/// If `reference` does not hold one value per point, or `output` one per point of `block`.
pub fn crosscheck<C: CommunicatorCollectives>(
    output: &Output<f64>,
    vector: usize,
    problem: &Problem,
    block: &Range<usize>,
    reference: &Values,
    grid: &Grid,
    comm: &C,
) -> Crosscheck {
    assert_eq!(
        reference.len(),
        problem.len(),
        "one reference value per point"
    );
    assert_eq!(
        output.ntargets(),
        block.len(),
        "one output per point of the block"
    );
    let phi = output.potential_of(vector);
    let grad = output.gradient_of(vector);
    let zero = Complex::new(0.0, 0.0);
    let (sums, boxes) = local_part(
        block.clone(),
        |i| {
            let j = i - block.start;
            let p = Complex::new(phi.re[j], phi.im.map_or(0.0, |im| im[j]));
            let g = grad.as_ref().map_or([zero; 3], |g| {
                std::array::from_fn(|a| Complex::new(g.re[j][a], g.im.map_or(0.0, |im| im[j][a])))
            });
            (p, g)
        },
        reference,
        &problem.points,
        grid,
    );
    let mut all = [0.0f64; 2];
    comm.all_reduce_into(
        &sums[..],
        &mut all[..],
        mpi::collective::SystemOperation::sum(),
    );
    let keys: Vec<u64> = boxes.keys().copied().collect();
    let maxima: Vec<f64> = keys.iter().flat_map(|k| boxes[k]).collect();
    let all_keys = all_gather(&keys, comm);
    let all_maxima = all_gather(&maxima, comm);
    let mut merged: HashMap<u64, [f64; 2]> = HashMap::new();
    for (k, e) in all_keys.iter().zip(all_maxima.as_chunks::<2>().0) {
        let entry = merged.entry(*k).or_insert([0.0, 0.0]);
        entry[0] = entry[0].max(e[0]);
        entry[1] = entry[1].max(e[1]);
    }
    finish(all, merged, reference, grid.level)
}

/// Every rank's `local`, concatenated in rank order, on every rank.
fn all_gather<T, C>(local: &[T], comm: &C) -> Vec<T>
where
    T: mpi::datatype::Equivalence + Copy + Default,
    C: CommunicatorCollectives,
{
    let size = comm.size() as usize;
    let length = local.len() as i32;
    let mut lengths = vec![0i32; size];
    comm.all_gather_into(&length, &mut lengths[..]);
    let offsets: Vec<i32> = lengths
        .iter()
        .scan(0, |acc, &c| {
            let o = *acc;
            *acc += c;
            Some(o)
        })
        .collect();
    let mut all = vec![T::default(); lengths.iter().sum::<i32>() as usize];
    comm.all_gather_varcount_into(
        local,
        &mut PartitionMut::new(&mut all[..], &lengths[..], &offsets[..]),
    );
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use nd_octree::PhysicalBox;

    use crate::helmholtz_accuracy::Case;
    use crate::helmholtz_workloads::Workload;
    use nd_fmm_exec::helmholtz::Degrees;

    fn unit_grid(level: usize) -> Grid {
        Grid {
            domain: Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap(),
            level,
        }
    }

    /// The driver's output bytes of `values` (sampled or full) as driver.f90 writes them.
    fn driver_bytes(magic: &[u8; 8], header: &[i64], values: &Values) -> Vec<u8> {
        let mut out = magic.to_vec();
        for h in header {
            out.extend_from_slice(&h.to_le_bytes());
        }
        for v in &values.potential {
            out.extend_from_slice(&v.re.to_le_bytes());
            out.extend_from_slice(&v.im.to_le_bytes());
        }
        for g in &values.gradient {
            for v in g {
                out.extend_from_slice(&v.re.to_le_bytes());
                out.extend_from_slice(&v.im.to_le_bytes());
            }
        }
        out
    }

    fn sample_values(n: usize) -> Values {
        let c = |i: usize, j: usize| Complex::new(i as f64 + 0.25 * j as f64, -(j as f64));
        Values {
            potential: (0..n).map(|i| c(i, 0)).collect(),
            gradient: (0..n).map(|i| [c(i, 1), c(i, 2), c(i, 3)]).collect(),
        }
    }

    #[test]
    fn problem_file_layout() {
        let mut case = Case::new(Workload::W1, (3.0, 0.0), Degrees::Tolerance(1e-3));
        case.samples = 5;
        let problem = Problem::new(&case);
        let bytes = problem_bytes(&problem, 0, 77);
        let n = problem.len();
        assert_eq!(&bytes[..8], PROBLEM_MAGIC);
        assert_eq!(
            [integer(&bytes, 8), integer(&bytes, 16), integer(&bytes, 24)],
            [n as i64, 5, 0]
        );
        assert_eq!(integer(&bytes, 32), 77);
        assert_eq!(bytes.len(), 40 + 24 * n + 16 * n + 8 * 5);
        let v = floats(&bytes[40..40 + 40 * n]);
        assert_eq!(&v[..3], &problem.points[0]);
        assert_eq!(v[3 * n], problem.re[0][0]);
        assert_eq!(v[3 * n + 1], problem.im[0][0]);
        let last = 40 + 40 * n;
        for (j, &s) in problem.sample.iter().enumerate() {
            assert_eq!(integer(&bytes, last + 8 * j), s as i64);
        }
    }

    #[test]
    fn output_files_round_trip() {
        let path = Path::new("test");
        let values = sample_values(3);
        let bytes = driver_bytes(SAMPLED_MAGIC, &[3, 0], &values);
        assert_eq!(parse_sampled(path, &bytes).unwrap(), values);
        // Extra targets after the samples are ignored.
        let mut with_targets = driver_bytes(SAMPLED_MAGIC, &[3, 1], &values);
        with_targets.extend(driver_bytes(b"........", &[], &sample_values(1))[8..].iter());
        assert_eq!(parse_sampled(path, &with_targets).unwrap(), values);
        let full = driver_bytes(FULL_MAGIC, &[3], &values);
        assert_eq!(parse_full(path, &full).unwrap(), values);
        assert_eq!(sampled_bytes(&values), bytes);
        assert_eq!(full_bytes(&values), full);
        assert!(parse_full(path, &bytes).is_err());
        assert!(parse_sampled(path, &full).is_err());
        assert!(parse_full(path, &full[..full.len() - 8]).is_err());
    }

    #[test]
    fn driver_line() {
        let log = "helmholtz: problem n=10 sampled=2 targets=0 seed=1 zk= 1 0\n\
                   helmholtz: token=1e-3 threads=6 repeat=1 seconds=  1.5000000000000000E-01 out=x\n\
                   helmholtz: token=1e-12 threads=6 repeat=2 seconds=  2.5E+00 out=y\n";
        assert_eq!(seconds(log, "1e-3"), Some((0.15, 6)));
        assert_eq!(seconds(log, "1e-12"), Some((2.5, 6)));
        assert_eq!(seconds(log, "1e-6"), None);
    }

    #[test]
    fn boxes_of_points() {
        assert_eq!(unit_grid(3).box_of([0.0, 0.0, 0.0]), 0);
        assert_eq!(
            unit_grid(3).box_of([1.0, 1.0, 1.0]),
            7 | (7 << 21) | (7 << 42)
        );
        assert_eq!(unit_grid(3).box_of([1.5, -0.5, 0.5]), 7 | (4 << 42));
        assert_eq!(
            unit_grid(2).box_of([0.5, 0.25, 0.999]),
            2 | (1 << 21) | (3 << 42)
        );
        assert_eq!(unit_grid(0).box_of([0.7, 0.2, 0.9]), 0);
    }

    #[test]
    fn crosscheck_norms_and_boxes() {
        let points = [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9],
            [0.8, 0.1, 0.1],
        ];
        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let reference = Values {
            potential: vec![one; 4],
            gradient: vec![[one, zero, zero]; 4],
        };
        let mut values = reference.clone();
        values.potential[1] += Complex::new(0.0, 0.3);
        values.potential[2] -= Complex::new(0.4, 0.0);
        values.gradient[3][2] = Complex::new(0.0, 0.5);
        let c = crosscheck_values(&values, &reference, &points, &unit_grid(1));
        // ‖Δφ‖ = 0.5, ‖φ‖ = 2; ‖Δ∇φ‖ = 0.5, ‖∇φ‖ = 2; both RMS are 1.
        assert!((c.errors[0] - 0.25).abs() < 1e-15);
        assert!((c.errors[1] - 0.25).abs() < 1e-15);
        assert_eq!(c.boxes.len(), 3);
        assert_eq!(c.boxes[0], (0, [0.3, 0.0]));
        assert_eq!(c.boxes[1], (1, [0.0, 0.5]));
        assert_eq!(c.boxes[2].1, [0.4, 0.0]);
        assert_eq!(c.worst(), [0.4, 0.5]);
    }

    #[test]
    fn sampled_errors_against_the_oracle() {
        let one = Complex::new(1.0, 0.0);
        let oracle = Oracle {
            potential: vec![vec![one, one]],
            gradient: vec![vec![[one; 3], [one; 3]]],
        };
        let mut values = Values {
            potential: vec![one, one],
            gradient: vec![[one; 3], [one; 3]],
        };
        assert_eq!(sample_errors(&values, &oracle, 0), [0.0, 0.0]);
        values.potential[0] = Complex::new(1.0, 2.0f64.sqrt());
        values.gradient[1][0] = Complex::new(1.0 + 6.0f64.sqrt(), 0.0);
        let e = sample_errors(&values, &oracle, 0);
        assert!((e[0] - 1.0).abs() < 1e-15);
        assert!((e[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn values_at_positions() {
        let v = sample_values(4);
        let w = v.at(&[2, 0]);
        assert_eq!(w.potential, vec![v.potential[2], v.potential[0]]);
        assert_eq!(w.gradient, vec![v.gradient[2], v.gradient[0]]);
    }
}
