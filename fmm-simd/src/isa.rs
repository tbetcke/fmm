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
/// [`Isa::Avx512`] joined in Phase 6 T6 (docs/design/optimisation.md §5); the enum is
/// `#[non_exhaustive]`, so a later ISA is not a breaking change either. It is selectable
/// wherever the CPU has it, but [`Isa::detect`] keeps [`Isa::Avx2`] on an AVX-512 machine
/// until Phase 6 T11 signs it off as the default (§5.3 there).
///
/// ```
/// use nd_fmm_simd::Isa;
///
/// assert!(Isa::Scalar.is_available());
/// assert!(Isa::available().any(|isa| isa == Isa::detect()));
/// assert_eq!(Isa::Avx2.lanes::<f32>(), 8);
/// assert_eq!(Isa::Avx512.lanes::<f64>(), 8);
/// assert_eq!(Isa::Neon.to_string(), "neon");
/// assert_ne!(Isa::detect(), Isa::Avx512);
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
    /// x86_64 AVX-512F, 512-bit vectors (Phase 6 T6). Detected at run time, together
    /// with AVX2 and FMA, which every AVX-512F CPU has and the compiler may use beside
    /// it. Never chosen by [`Isa::detect`] (yet); select it explicitly.
    Avx512,
}

/// Every variant, in the order [`Isa::all`] and [`Isa::available`] report them.
const ALL: [Isa; 4] = [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512];

/// The CPU features the ISAs need, as detected on this machine or, in the tests, forced:
/// the dispatch rule is a function of them alone (docs/design/optimisation.md §5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Features {
    /// aarch64 Advanced SIMD.
    pub(crate) neon: bool,
    /// x86_64 AVX2 and FMA, both.
    pub(crate) avx2_fma: bool,
    /// x86_64 AVX-512F.
    pub(crate) avx512f: bool,
}

impl Features {
    /// The features of this machine: NEON from the target, the x86_64 ones by
    /// `is_x86_feature_detected!` (cached by `std`, so cheap to repeat).
    pub(crate) fn host() -> Self {
        Self {
            neon: cfg!(all(target_arch = "aarch64", target_feature = "neon")),
            avx2_fma: avx2_fma_detected(),
            avx512f: avx512f_detected(),
        }
    }

    /// Whether a CPU with these features can run `isa`. AVX-512 needs AVX2 and FMA as
    /// well: its entry points enable `avx512f`, which implies both to the compiler.
    pub(crate) fn runs(self, isa: Isa) -> bool {
        match isa {
            Isa::Scalar => true,
            Isa::Neon => self.neon,
            Isa::Avx2 => self.avx2_fma,
            Isa::Avx512 => self.avx512f && self.avx2_fma,
        }
    }

    /// The ISA [`Isa::detect`] picks with these features: AVX2 + FMA, else NEON, else
    /// scalar. Never [`Isa::Avx512`]: the default stays AVX2 on an AVX-512 machine until
    /// Phase 6 T11 signs it off (docs/design/optimisation.md §5.3).
    pub(crate) fn detect(self) -> Isa {
        if self.runs(Isa::Avx2) {
            Isa::Avx2
        } else if self.runs(Isa::Neon) {
            Isa::Neon
        } else {
            Isa::Scalar
        }
    }
}

impl Isa {
    /// The default ISA of this machine: [`Isa::Neon`] on aarch64, [`Isa::Avx2`] on
    /// x86_64 when the CPU has both AVX2 and FMA, else [`Isa::Scalar`].
    ///
    /// An AVX-512 machine gets [`Isa::Avx2`]: [`Isa::Avx512`] changes the bits of every
    /// output (another inverse-square-root estimate), and Cascade Lake lowers a core's
    /// clock while it runs 512-bit code, so it becomes the default only where Phase 6
    /// measures it faster inside the FMM, at T11's sign-off
    /// (docs/design/optimisation.md §5.3).
    ///
    /// Detection reads the CPU, never an environment variable. On x86_64 `std` caches
    /// the result of the CPUID query, so repeated calls are cheap.
    pub fn detect() -> Isa {
        Features::host().detect()
    }

    /// Whether this machine can run the ISA.
    ///
    /// Always true for [`Isa::Scalar`], and always false for an ISA of another
    /// architecture.
    pub fn is_available(self) -> bool {
        Features::host().runs(self)
    }

    /// Every ISA, available or not: [`Isa::Scalar`], [`Isa::Neon`], [`Isa::Avx2`],
    /// [`Isa::Avx512`].
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
    /// [`Isa::Avx2`]; 16 and 8 for [`Isa::Avx512`] (docs/design/simd-p2p.md §3,
    /// requirement 7).
    ///
    /// This is a property of the ISA, whether or not this machine can run it.
    pub fn lanes<T: SimdScalar>(self) -> usize {
        let register_bytes = match self {
            Isa::Scalar => return 1,
            Isa::Neon => 16,
            Isa::Avx2 => 32,
            Isa::Avx512 => 64,
        };
        register_bytes / size_of::<T>()
    }

    /// The lower-case name that [`Display`](fmt::Display) prints.
    fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Neon => "neon",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }
}

/// Prints the lower-case name: `scalar`, `neon`, `avx2` or `avx512`.
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

/// Whether the CPU has AVX-512F (and the operating system saves its registers, which
/// `is_x86_feature_detected!` checks too).
#[cfg(target_arch = "x86_64")]
fn avx512f_detected() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

/// AVX-512 exists only on x86_64.
#[cfg(not(target_arch = "x86_64"))]
fn avx512f_detected() -> bool {
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

#[cfg(test)]
mod tests {
    use super::{Features, Isa};

    /// Every combination of the three features.
    fn every_feature_set() -> impl Iterator<Item = Features> {
        (0..8u8).map(|bits| Features {
            neon: bits & 1 != 0,
            avx2_fma: bits & 2 != 0,
            avx512f: bits & 4 != 0,
        })
    }

    /// The dispatch rule with forced feature sets (docs/design/optimisation.md §5.4):
    /// detection picks an ISA the set runs, never AVX-512, and AVX2 wherever the set has
    /// it, AVX-512F included; AVX-512 runs only with AVX-512F, AVX2 and FMA.
    #[test]
    fn dispatch_with_forced_features() {
        for features in every_feature_set() {
            let detected = features.detect();
            assert!(features.runs(detected), "{features:?}: {detected}");
            assert_ne!(detected, Isa::Avx512, "{features:?}");
            assert!(features.runs(Isa::Scalar));
            assert_eq!(
                features.runs(Isa::Avx512),
                features.avx512f && features.avx2_fma,
                "{features:?}"
            );
            if features.avx2_fma {
                assert_eq!(detected, Isa::Avx2, "{features:?}");
            }
        }
        let cascade_lake = Features {
            neon: false,
            avx2_fma: true,
            avx512f: true,
        };
        assert_eq!(cascade_lake.detect(), Isa::Avx2);
        assert!(cascade_lake.runs(Isa::Avx512));
        let none = Features {
            neon: false,
            avx2_fma: false,
            avx512f: false,
        };
        assert_eq!(none.detect(), Isa::Scalar);
        assert!(!none.runs(Isa::Avx512));
        println!(
            "dispatch_with_forced_features: this machine {:?}, detected {}",
            Features::host(),
            Isa::detect()
        );
    }
}
