//! Acceptance tests for `MatrixKernel::accumulate_columns` (Phase 6 T2, C6.6;
//! docs/design/optimisation.md §2.3, §2.11) on every ISA this machine offers, in f32 and
//! f64, in both multiply–add modes.
//!
//! Error measures:
//! - *Bits*: exact equality of every value of C, compared by its bit pattern. The
//!   separate mode equals the matrix–vector loop of `nd_fmm_tables::MatrixSet::apply`
//!   (restated here as [`naive`]) on every ISA; the fused mode equals the same loop with
//!   `mul_add` on every ISA. Every ISA equals the scalar path in both modes.
//! - *Fused against separate*: per element, |fused − separate| relative to
//!   |c₀| + Σₖ |B[k, b_j] A[r, k]|, at most 2 n u_T (each chain's error is at most
//!   γ_n ≈ n u_T of that sum; optimisation.md §2.3).
//!
//! The degrees p = 0–20 (n = (p + 1)²) and the column counts 1–64 run in full in the
//! ignored sweep, for a release run (`cargo test -p nd-fmm-simd --release -- --ignored
//! --show-output`); the debug run checks a strided subset. Every test prints the ISAs it
//! ran.

use nd_fmm_simd::{Isa, MatrixKernel, MulAdd, SimdScalar};

/// The degrees and column counts of the debug run.
const DEBUG_DEGREES: [usize; 7] = [0, 1, 2, 3, 5, 8, 12];
const DEBUG_COLUMNS: [usize; 9] = [1, 2, 3, 4, 5, 7, 8, 9, 33];

/// SplitMix64, seeded.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 2f64.powi(-53)
    }

    /// A value of either sign with a random significand, of magnitude in [2⁻⁶, 2⁶).
    fn value<T: SimdScalar>(&mut self) -> T {
        let magnitude = 2f64.powf(12.0 * self.unit() - 6.0);
        let sign = if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        T::from_f64(sign * magnitude)
    }

    /// Below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// A problem: A (n × n), B with `nb` columns read by `m` offsets in random order with
/// repeats, C with gaps between its `m` ascending columns, and C's initial values.
struct Problem<T> {
    n: usize,
    a: Vec<T>,
    b: Vec<T>,
    b_offsets: Vec<usize>,
    c: Vec<T>,
    c_offsets: Vec<usize>,
}

impl<T: SimdScalar> Problem<T> {
    fn new(p: usize, m: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let n = (p + 1) * (p + 1);
        let nb = m.div_ceil(2) + 1;
        let a = (0..n * n).map(|_| rng.value()).collect();
        let b = (0..nb * n + 3).map(|_| rng.value()).collect();
        // Columns of B at any position (not only multiples of n), some read twice.
        let b_offsets = (0..m).map(|_| rng.below(nb * n + 3 - n + 1)).collect();
        let mut c_offsets = Vec::with_capacity(m);
        let mut next = rng.below(3);
        for _ in 0..m {
            c_offsets.push(next);
            next += n + rng.below(3);
        }
        let c = (0..next + rng.below(3)).map(|_| rng.value()).collect();
        Self {
            n,
            a,
            b,
            b_offsets,
            c,
            c_offsets,
        }
    }

    /// C after the kernel.
    fn run(&self, kernel: &MatrixKernel<T>) -> Vec<T> {
        let mut c = self.c.clone();
        let mut pack = vec![T::zero(); MatrixKernel::<T>::pack_len(self.n)];
        kernel.accumulate_columns(
            self.n,
            &self.a,
            &self.b,
            &self.b_offsets,
            &mut c,
            &self.c_offsets,
            &mut pack,
        );
        c
    }
}

/// The matrix–vector loop of `MatrixSet::apply`, column by column: for k ascending,
/// c[r] ← c[r] + b[k] · a[r, k] (separate) or `b[k].mul_add(a[r, k], c[r])` (fused).
fn naive<T: SimdScalar>(problem: &Problem<T>, mul_add: MulAdd) -> Vec<T> {
    let n = problem.n;
    let mut c = problem.c.clone();
    for (&bo, &co) in problem.b_offsets.iter().zip(&problem.c_offsets) {
        let x = &problem.b[bo..bo + n];
        let y = &mut c[co..co + n];
        for (&xk, column) in x.iter().zip(problem.a.chunks_exact(n)) {
            for (yr, &ark) in y.iter_mut().zip(column) {
                *yr = match mul_add {
                    MulAdd::Separate => *yr + xk * ark,
                    MulAdd::Fused => xk.mul_add(ark, *yr),
                };
            }
        }
    }
    c
}

/// The bits of every value.
fn bits<T: SimdScalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|&v| v.to_f64().to_bits()).collect()
}

/// The ISAs this machine cannot run (those of the other architecture, and on x86_64 any
/// whose features the CPU lacks, such as AVX-512 on most machines), for the line every
/// test prints: their tests skip, they do not fail.
fn unavailable() -> String {
    let names: Vec<String> = Isa::all()
        .filter(|isa| !isa.is_available())
        .map(|isa| isa.to_string())
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// The kernels of every available ISA in `mul_add`, after printing them under the test's
/// name.
fn kernels<T: SimdScalar>(test: &str, mul_add: MulAdd) -> Vec<MatrixKernel<T>> {
    let kernels: Vec<MatrixKernel<T>> = Isa::available()
        .map(|isa| MatrixKernel::new(isa, mul_add).expect("an available ISA"))
        .collect();
    let names: Vec<String> = kernels.iter().map(|k| k.isa().to_string()).collect();
    println!(
        "{test} ({}, {mul_add}): ISAs run: {}; not available here: {}",
        std::any::type_name::<T>(),
        names.join(", "),
        unavailable()
    );
    kernels
}

/// Every ISA against the naive loop, bit for bit, over `degrees` × `columns`.
fn every_isa_bit_for_bit<T: SimdScalar>(test: &str, degrees: &[usize], columns: &[usize]) {
    for mul_add in [MulAdd::Separate, MulAdd::Fused] {
        let kernels = kernels::<T>(test, mul_add);
        for &p in degrees {
            for &m in columns {
                let problem = Problem::<T>::new(p, m, (p * 1000 + m) as u64);
                let expected = bits(&naive(&problem, mul_add));
                for kernel in &kernels {
                    assert_eq!(
                        bits(&problem.run(kernel)),
                        expected,
                        "{} {mul_add} p = {p}, {m} columns",
                        kernel.isa()
                    );
                }
            }
        }
    }
}

#[test]
fn every_isa_equals_the_matrix_vector_loop_f32() {
    every_isa_bit_for_bit::<f32>("matrix bits", &DEBUG_DEGREES, &DEBUG_COLUMNS);
}

#[test]
fn every_isa_equals_the_matrix_vector_loop_f64() {
    every_isa_bit_for_bit::<f64>("matrix bits", &DEBUG_DEGREES, &DEBUG_COLUMNS);
}

#[test]
#[ignore = "release sweep: p = 0–20 and 1–64 columns"]
fn every_isa_equals_the_matrix_vector_loop_sweep() {
    let degrees: Vec<usize> = (0..=20).collect();
    let columns: Vec<usize> = (1..=64).collect();
    every_isa_bit_for_bit::<f32>("matrix bits sweep", &degrees, &columns);
    every_isa_bit_for_bit::<f64>("matrix bits sweep", &degrees, &columns);
}

/// Splitting the columns over two calls, or reversing their order (with C's columns
/// still ascending), changes no bit: each element depends on its own column only.
fn columns_are_independent<T: SimdScalar>() {
    for mul_add in [MulAdd::Separate, MulAdd::Fused] {
        for kernel in kernels::<T>("matrix column independence", mul_add) {
            for (p, m) in [(3, 9), (8, 13), (6, 4)] {
                let problem = Problem::<T>::new(p, m, 7 + p as u64);
                let whole = bits(&problem.run(&kernel));
                let mut pack = vec![T::zero(); MatrixKernel::<T>::pack_len(problem.n)];
                for split in 0..=m {
                    let mut c = problem.c.clone();
                    for range in [0..split, split..m] {
                        kernel.accumulate_columns(
                            problem.n,
                            &problem.a,
                            &problem.b,
                            &problem.b_offsets[range.clone()],
                            &mut c,
                            &problem.c_offsets[range],
                            &mut pack,
                        );
                    }
                    assert_eq!(bits(&c), whole, "{} split at {split}", kernel.isa());
                }
                // One column per call, last first.
                let mut c = problem.c.clone();
                for j in (0..m).rev() {
                    kernel.accumulate_columns(
                        problem.n,
                        &problem.a,
                        &problem.b,
                        &problem.b_offsets[j..=j],
                        &mut c,
                        &problem.c_offsets[j..=j],
                        &mut pack,
                    );
                }
                assert_eq!(bits(&c), whole, "{} column by column", kernel.isa());
            }
        }
    }
}

#[test]
fn columns_are_independent_f32() {
    columns_are_independent::<f32>();
}

#[test]
fn columns_are_independent_f64() {
    columns_are_independent::<f64>();
}

/// The fused mode within 2 n u_T of the separate mode, relative to |c₀| + Σₖ |b a|.
fn fused_within_bound<T: SimdScalar>(unit: f64) {
    let separate = MatrixKernel::<T>::new(Isa::Scalar, MulAdd::Separate).unwrap();
    let mut worst = 0.0f64;
    for kernel in kernels::<T>("matrix fused bound", MulAdd::Fused) {
        for p in [0, 3, 8, 12, 20] {
            let problem = Problem::<T>::new(p, 6, 99 + p as u64);
            let (fused, exact) = (problem.run(&kernel), problem.run(&separate));
            let n = problem.n;
            for (&bo, &co) in problem.b_offsets.iter().zip(&problem.c_offsets) {
                for r in 0..n {
                    let scale = problem.c[co + r].to_f64().abs()
                        + (0..n)
                            .map(|k| {
                                (problem.b[bo + k].to_f64() * problem.a[r + k * n].to_f64()).abs()
                            })
                            .sum::<f64>();
                    let error = (fused[co + r].to_f64() - exact[co + r].to_f64()).abs() / scale;
                    worst = worst.max(error / (n as f64 * unit));
                    assert!(
                        error <= 2.0 * n as f64 * unit,
                        "{} p = {p}, row {r}: {error:e} > 2 n u",
                        kernel.isa()
                    );
                }
            }
        }
    }
    println!(
        "matrix fused bound ({}): largest |fused − separate| / (n u Σ|terms|) = {worst:.3}",
        std::any::type_name::<T>()
    );
}

#[test]
fn fused_within_bound_f32() {
    fused_within_bound::<f32>(2f64.powi(-24));
}

#[test]
fn fused_within_bound_f64() {
    fused_within_bound::<f64>(2f64.powi(-53));
}

#[test]
fn modes_and_isas_are_reported() {
    for isa in Isa::all() {
        for mul_add in [MulAdd::Separate, MulAdd::Fused] {
            match MatrixKernel::<f64>::new(isa, mul_add) {
                Ok(kernel) => {
                    assert!(isa.is_available());
                    assert_eq!((kernel.isa(), kernel.mul_add()), (isa, mul_add));
                }
                Err(error) => {
                    assert!(!isa.is_available());
                    assert_eq!(error.isa, isa);
                }
            }
        }
    }
    assert_eq!(
        MatrixKernel::<f32>::detect(MulAdd::Fused).isa(),
        Isa::detect()
    );
    assert_eq!("separate".parse(), Ok(MulAdd::Separate));
    assert!("fma".parse::<MulAdd>().is_err());
    assert_eq!(MatrixKernel::<f64>::pack_len(81), 324);
}

#[test]
fn no_columns_or_order_zero_is_a_no_op() {
    let kernel = MatrixKernel::<f64>::detect(MulAdd::Separate);
    let mut c = [1.0, 2.0];
    kernel.accumulate_columns(1, &[3.0], &[5.0], &[], &mut c, &[], &mut [0.0; 4]);
    kernel.accumulate_columns(0, &[], &[], &[0, 0], &mut c, &[0, 0], &mut []);
    assert_eq!(c, [1.0, 2.0]);
}

/// Runs `accumulate_columns` with order 2 and the given lists and lengths.
fn call(a: usize, b_offsets: &[usize], c_offsets: &[usize], c: usize, pack: usize) {
    let kernel = MatrixKernel::<f64>::detect(MulAdd::Separate);
    kernel.accumulate_columns(
        2,
        &vec![1.0; a],
        &[1.0; 6],
        b_offsets,
        &mut vec![0.0; c],
        c_offsets,
        &mut vec![0.0; pack],
    );
}

#[test]
#[should_panic(expected = "`a` must hold n² = 4 values, got 3")]
fn panics_on_a_wrong_matrix() {
    call(3, &[0], &[0], 2, 8);
}

#[test]
#[should_panic(expected = "`b_offsets` and `c_offsets` must have the same length")]
fn panics_on_lists_of_different_lengths() {
    call(4, &[0, 2], &[0], 4, 8);
}

#[test]
#[should_panic(expected = "the column of `b` at 5 ends beyond its 6 values")]
fn panics_on_a_column_of_b_out_of_bounds() {
    call(4, &[5], &[0], 2, 8);
}

#[test]
#[should_panic(expected = "the column of `c` at 3 ends beyond its 4 values")]
fn panics_on_a_column_of_c_out_of_bounds() {
    call(4, &[0], &[3], 4, 8);
}

#[test]
#[should_panic(expected = "the columns of `c` must be ascending and disjoint: 1 after")]
fn panics_on_overlapping_columns_of_c() {
    call(4, &[0, 2], &[0, 1], 4, 8);
}

#[test]
#[should_panic(expected = "the columns of `c` must be ascending and disjoint: 0 after")]
fn panics_on_descending_columns_of_c() {
    call(4, &[0, 2], &[2, 0], 4, 8);
}

#[test]
#[should_panic(expected = "`pack` must hold at least 8 values, got 7")]
fn panics_on_a_short_packing_buffer() {
    call(4, &[0], &[0], 2, 7);
}
