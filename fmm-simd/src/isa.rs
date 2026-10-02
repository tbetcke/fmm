//! Instruction sets, their detection and the error for an unavailable one
//! (docs/design/simd-p2p.md §5.2).

use std::fmt;

use crate::SimdScalar;

/// An instruction set with a P2P path.
///
/// Every variant exists on every target, so that settings and command lines mean the
/// same everywhere; [`Isa::is_available`] says whether this machine can run it. The
/// code of an architecture-specific ISA is compiled only for its architecture.
///
/// AVX-512 is deferred until hardware to test and time it is available
/// (docs/design/simd-p2p.md §4.7). The enum is `#[non_exhaustive]` so that
/// `Isa::Avx512` can be added later without a breaking change; an AVX-512 machine runs
/// [`Isa::Avx2`] until then.
///
/// ```
/// use nd_fmm_simd::Isa;
///
/// assert!(Isa::Scalar.is_available());
/// assert!(Isa::available().any(|isa| isa == Isa::detect()));
/// assert_eq!(Isa::Avx2.lanes::<f32>(), 8);
/// assert_eq!(Isa::Neon.to_string(), "neon");
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// The portable scalar path: a plain loop with `sqrt` and division. Available
    /// everywhere; the yardstick of the vector paths.
    Scalar,
    /// aarch64 Advanced SIMD (NEON), 128-bit vectors. Part of the base architecture,
    /// so it is available on every aarch64 CPU without detection.
    Neon,
    /// x86_64 AVX2 together with FMA, 256-bit vectors. Detected at run time.
    Avx2,
}

/// Every variant, in the order [`Isa::all`] and [`Isa::available`] report them.
const ALL: [Isa; 3] = [Isa::Scalar, Isa::Neon, Isa::Avx2];

impl Isa {
    /// The widest ISA this machine can run: [`Isa::Neon`] on aarch64, [`Isa::Avx2`] on
    /// x86_64 when the CPU has both AVX2 and FMA, else [`Isa::Scalar`].
    ///
    /// Detection reads the CPU, never an environment variable. On x86_64 `std` caches
    /// the result of the CPUID query, so repeated calls are cheap.
    pub fn detect() -> Isa {
        if Isa::Avx2.is_available() {
            Isa::Avx2
        } else if Isa::Neon.is_available() {
            Isa::Neon
        } else {
            Isa::Scalar
        }
    }

    /// Whether this machine can run the ISA.
    ///
    /// Always true for [`Isa::Scalar`], and always false for an ISA of another
    /// architecture.
    pub fn is_available(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Neon => cfg!(all(target_arch = "aarch64", target_feature = "neon")),
            Isa::Avx2 => avx2_fma_detected(),
        }
    }

    /// Every ISA, available or not: [`Isa::Scalar`], [`Isa::Neon`], [`Isa::Avx2`].
    pub fn all() -> impl Iterator<Item = Isa> {
        ALL.into_iter()
    }

    /// The ISAs this machine can run, [`Isa::Scalar`] first, then in the order of
    /// [`Isa::all`]. Always contains [`Isa::Scalar`] and [`Isa::detect`].
    pub fn available() -> impl Iterator<Item = Isa> {
        Isa::all().filter(|isa| isa.is_available())
    }

    /// The number of values of `T` in one vector register of the ISA: 1 for
    /// [`Isa::Scalar`]; 4 (f32) and 2 (f64) for [`Isa::Neon`]; 8 and 4 for
    /// [`Isa::Avx2`] (docs/design/simd-p2p.md §3, requirement 7).
    ///
    /// This is a property of the ISA, whether or not this machine can run it.
    pub fn lanes<T: SimdScalar>(self) -> usize {
        let register_bytes = match self {
            Isa::Scalar => return 1,
            Isa::Neon => 16,
            Isa::Avx2 => 32,
        };
        register_bytes / size_of::<T>()
    }

    /// The lower-case name that [`Display`](fmt::Display) prints.
    fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Neon => "neon",
            Isa::Avx2 => "avx2",
        }
    }
}

/// Prints the lower-case name: `scalar`, `neon` or `avx2`.
impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Whether the CPU has both AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
fn avx2_fma_detected() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// AVX2 exists only on x86_64.
#[cfg(not(target_arch = "x86_64"))]
fn avx2_fma_detected() -> bool {
    false
}

/// The error of [`P2pKernel::new`](crate::P2pKernel::new) for an ISA this machine
/// cannot run.
///
/// ```
/// use nd_fmm_simd::{Isa, P2pKernel};
///
/// for isa in Isa::all() {
///     match P2pKernel::<f64>::new(isa) {
///         Ok(kernel) => assert_eq!(kernel.isa(), isa),
///         Err(error) => assert_eq!(error.isa, isa),
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("instruction set `{isa}` is not available on this machine")]
pub struct IsaUnavailable {
    /// The ISA that was asked for.
    pub isa: Isa,
}
