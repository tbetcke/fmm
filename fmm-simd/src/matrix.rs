//! The column-accumulation kernel bound to one ISA: the product of the host's batched
//! M2L (docs/design/optimisation.md §2.2, §2.3, §2.11; Phase 6 T2, C6.6).

use core::fmt;
use core::str::FromStr;
use std::marker::PhantomData;

use crate::arch::{AvailableIsa, matrix::COLUMNS};
use crate::{Isa, IsaUnavailable, SimdScalar};

/// How [`MatrixKernel`] forms each step c ← c + b a of its chains.
///
/// | Mode | Step | Bits |
/// | --- | --- | --- |
/// | [`Separate`](Self::Separate), the default | the product, rounded, then the sum, rounded | those of `nd_fmm_tables::MatrixSet::apply` |
/// | [`Fused`](Self::Fused) | one fused multiply–add, rounded once | within n u relative of the separate form's, per element |
///
/// Both are deterministic, and the same on every ISA: IEEE 754 multiplication, addition
/// and fma are correctly rounded everywhere, and the scalar path forms the fused step
/// with `mul_add`. The text form, for command lines, is `separate` or `fused`:
///
/// ```
/// use nd_fmm_simd::MulAdd;
///
/// assert_eq!(MulAdd::default(), MulAdd::Separate);
/// assert_eq!("fused".parse(), Ok(MulAdd::Fused));
/// assert_eq!(MulAdd::Separate.to_string(), "separate");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MulAdd {
    /// c ← c + b · a with the product and the sum rounded apart (no fma): the chain of
    /// `MatrixSet::apply`, so the result is that of the matrix–vector loop bit for bit.
    #[default]
    Separate,
    /// c ← fma(a, b, c), one rounding per step (the device's form). Each element differs
    /// from the separate form's by at most the rounding errors of its n partial sums:
    /// relative to |A| |B| (the sum of the absolute terms) at most about n u_T.
    Fused,
}

/// Prints `separate` or `fused`.
impl fmt::Display for MulAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Separate => "separate",
            Self::Fused => "fused",
        })
    }
}

/// The error of parsing a [`MulAdd`] from text it does not name.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{text}` is not a multiply–add mode; expected `separate` or `fused`")]
pub struct UnknownMulAdd {
    /// The text that was parsed.
    pub text: String,
}

/// Parses the [`Display`](fmt::Display) form: `separate` or `fused`.
impl FromStr for MulAdd {
    type Err = UnknownMulAdd;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "separate" => Ok(Self::Separate),
            "fused" => Ok(Self::Fused),
            _ => Err(UnknownMulAdd { text: s.to_owned() }),
        }
    }
}

/// The column-accumulation kernel bound to one available [`Isa`] and one [`MulAdd`]
/// mode: C[:, c_j] += A · B[:, b_j] for lists of columns, the product of the host's
/// batched M2L (docs/design/optimisation.md §2.2).
///
/// [`accumulate_columns`](Self::accumulate_columns) gives every element of C the chain of
/// `nd_fmm_tables::MatrixSet::apply` (k ascending, one rounded multiply and one rounded
/// add per step) in the [`Separate`](MulAdd::Separate) mode, so a product over any set of
/// columns equals the matrix–vector products of its columns one by one, bit for bit, on
/// every ISA. The kernel is stateless: it is `Copy`, `Send` and `Sync`.
///
/// ```
/// use nd_fmm_simd::{Isa, MatrixKernel, MulAdd};
///
/// let kernel = MatrixKernel::<f64>::new(Isa::detect(), MulAdd::Separate).unwrap();
/// let a = [1.0, 2.0, 3.0, 4.0]; // A = [1 3; 2 4], column-major
/// let b = [1.0, 1.0, 0.0, 1.0]; // two columns: (1, 1) and (0, 1)
/// let mut c = [10.0, 20.0, 0.0, 0.0, 0.0];
/// let mut pack = vec![0.0; MatrixKernel::<f64>::pack_len(2)];
/// // C[:, 0] += A B[:, 0], and C[:, 3..5] += A B[:, 2..4].
/// kernel.accumulate_columns(2, &a, &b, &[0, 2], &mut c, &[0, 3], &mut pack);
/// assert_eq!(c, [14.0, 26.0, 0.0, 3.0, 4.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MatrixKernel<T: SimdScalar> {
    /// The ISA, checked to be available by every constructor.
    isa: AvailableIsa,
    /// The multiply–add mode.
    mul_add: MulAdd,
    /// The precision.
    precision: PhantomData<T>,
}

impl<T: SimdScalar> MatrixKernel<T> {
    /// The kernel on `isa` in the mode `mul_add`.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] if this machine cannot run `isa` ([`Isa::is_available`]).
    pub fn new(isa: Isa, mul_add: MulAdd) -> Result<Self, IsaUnavailable> {
        Ok(Self {
            isa: AvailableIsa::new(isa)?,
            mul_add,
            precision: PhantomData,
        })
    }

    /// The kernel on the default ISA of this machine, [`Isa::detect`], in the mode `mul_add`.
    pub fn detect(mul_add: MulAdd) -> Self {
        Self::new(Isa::detect(), mul_add).expect("`Isa::detect` returns an available ISA")
    }

    /// The ISA the kernel runs on.
    pub fn isa(&self) -> Isa {
        self.isa.isa()
    }

    /// The multiply–add mode.
    pub fn mul_add(&self) -> MulAdd {
        self.mul_add
    }

    /// The length of the packing buffer that
    /// [`accumulate_columns`](Self::accumulate_columns) needs for matrices of order `n`:
    /// 4 n values (four source columns packed together), the same on every ISA.
    pub fn pack_len(n: usize) -> usize {
        COLUMNS * n
    }

    /// Adds A · B[:, b_j] into C[:, c_j] for every j: with A the n × n matrix `a`,
    /// column-major, column b_j of B the n values `b[b_offsets[j]..][..n]` and column c_j
    /// of C the n values `c[c_offsets[j]..][..n]`.
    ///
    /// - **Order.** Every element C[r, c_j] receives c ← c + B[k, b_j] · A[r, k] for k =
    ///   0, 1, …, n − 1 in turn, starting from its value in C: in the
    ///   [`Separate`](MulAdd::Separate) mode the product and the sum rounded apart, the
    ///   chain of `MatrixSet::apply`; in the [`Fused`](MulAdd::Fused) mode one fma per
    ///   step. An element depends on its own row of A and column of B only, so the result
    ///   does not depend on the ISA, on which columns share a call or on their order, bit
    ///   for bit; a column may be read by several j.
    /// - **Work.** 2 n² flops per column, A read once per four columns; `pack` holds the
    ///   packed columns ([`pack_len`](Self::pack_len)). The kernel neither allocates nor
    ///   spawns threads, and writes only into `c` and `pack`.
    ///
    /// Distinct C columns, ascending: `c_offsets[j] + n ≤ c_offsets[j + 1]`.
    ///
    /// # Panics
    ///
    /// If `a` does not hold n² values, the offset lists differ in length, a column of B
    /// or C lies outside its slice, the C offsets are not ascending by at least n, or
    /// `pack` holds fewer than [`pack_len`](Self::pack_len)`(n)` values.
    #[expect(
        clippy::too_many_arguments,
        reason = "a matrix, two column lists over two slices, and the packing buffer"
    )]
    pub fn accumulate_columns(
        &self,
        n: usize,
        a: &[T],
        b: &[T],
        b_offsets: &[usize],
        c: &mut [T],
        c_offsets: &[usize],
        pack: &mut [T],
    ) {
        check_shapes(
            n,
            a.len(),
            b.len(),
            b_offsets,
            c.len(),
            c_offsets,
            pack.len(),
        );
        if n == 0 || b_offsets.is_empty() {
            return;
        }
        T::accumulate(
            self.isa,
            self.mul_add,
            n,
            a,
            b,
            b_offsets,
            c,
            c_offsets,
            pack,
        );
    }
}

/// Panics unless the shapes of [`MatrixKernel::accumulate_columns`] hold.
fn check_shapes(
    n: usize,
    a: usize,
    b: usize,
    b_offsets: &[usize],
    c: usize,
    c_offsets: &[usize],
    pack: usize,
) {
    assert_eq!(a, n * n, "`a` must hold n² = {} values, got {a}", n * n);
    assert_eq!(
        b_offsets.len(),
        c_offsets.len(),
        "`b_offsets` and `c_offsets` must have the same length"
    );
    assert!(
        pack >= COLUMNS * n,
        "`pack` must hold at least {} values, got {pack}",
        COLUMNS * n
    );
    for &offset in b_offsets {
        assert!(
            offset.checked_add(n).is_some_and(|end| end <= b),
            "the column of `b` at {offset} ends beyond its {b} values"
        );
    }
    let mut next = 0;
    for &offset in c_offsets {
        assert!(
            offset >= next,
            "the columns of `c` must be ascending and disjoint: {offset} after a column \
             ending at {next}"
        );
        next = offset
            .checked_add(n)
            .filter(|&end| end <= c)
            .unwrap_or_else(|| panic!("the column of `c` at {offset} ends beyond its {c} values"));
    }
}
