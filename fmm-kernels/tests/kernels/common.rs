//! Shared devices, the backend report of every test, and host data.

#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
use nd_fmm_kernels::Device;
use nd_fmm_kernels::{BackendKind, DeviceFloat};

/// One device per backend and process: CubeCL caches compiled kernels per process
/// only (device-path.md §13.3, F23), and a test holds the device for its whole body.
#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
static DEVICES: [OnceLock<Mutex<Device>>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];

/// The shared device of `kind`, opened on first use.
///
/// # Panics
///
/// If the device cannot be opened: a test asked for a backend, so a backend that does
/// not come up fails the test rather than passing it.
#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
pub fn device(kind: BackendKind) -> MutexGuard<'static, Device> {
    let slot = &DEVICES[BackendKind::ALL.iter().position(|&k| k == kind).unwrap()];
    slot.get_or_init(|| {
        Mutex::new(Device::open(kind).unwrap_or_else(|e| panic!("cannot open {kind}: {e}")))
    })
    .lock()
    .unwrap_or_else(PoisonError::into_inner)
}

/// Runs `body` on the shared device of `kind`, printing the device before and the
/// backends run and not run after.
#[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
pub fn run(kind: BackendKind, test: &str, body: fn(&mut Device)) {
    let mut device = device(kind);
    println!("{test}: on {}", device.info());
    body(&mut device);
    println!("{test}: {}", backends_line(&[kind]));
}

/// "backends run: …; not run: …", with the reason for each backend not run.
pub fn backends_line(run: &[BackendKind]) -> String {
    let ran: Vec<_> = run.iter().map(|k| k.name()).collect();
    let not_run: Vec<String> = BackendKind::ALL
        .into_iter()
        .filter(|k| !run.contains(k))
        .map(|k| {
            let why = match (k, k.is_compiled()) {
                (_, false) => "not compiled",
                (BackendKind::Metal | BackendKind::Cuda, true) => "in its own ignored test",
                (BackendKind::Cpu, true) => "in its own test",
            };
            format!("{k} ({why})")
        })
        .collect();
    format!(
        "backends run: {}; not run: {}",
        if ran.is_empty() {
            "none".to_string()
        } else {
            ran.join(", ")
        },
        not_run.join(", ")
    )
}

/// Test functions on the CPU runtime (`cpu: …`, plain tests), on Metal (`metal: …`,
/// ignored, run by hand outside the sandbox) or on CUDA (`cuda: …`, ignored, run by hand
/// on locust). Each names a `fn(&mut Device)` of the enclosing module, which is
/// referenced even when its backend is not compiled in.
macro_rules! tests_on {
    (cpu: $($name:ident),* $(,)?) => {
        const _: &[fn(&mut nd_fmm_kernels::Device)] = &[$($name),*];
        #[cfg(feature = "cpu")]
        mod cpu {
            $(
                #[test]
                fn $name() {
                    crate::common::run(
                        nd_fmm_kernels::BackendKind::Cpu,
                        concat!(module_path!(), "::", stringify!($name)),
                        super::$name,
                    );
                }
            )*
        }
    };
    (metal: $($name:ident),* $(,)?) => {
        const _: &[fn(&mut nd_fmm_kernels::Device)] = &[$($name),*];
        #[cfg(feature = "metal")]
        mod metal {
            $(
                #[test]
                #[ignore = "Metal: run by hand, outside the sandbox"]
                fn $name() {
                    crate::common::run(
                        nd_fmm_kernels::BackendKind::Metal,
                        concat!(module_path!(), "::", stringify!($name)),
                        super::$name,
                    );
                }
            )*
        }
    };
    (cuda: $($name:ident),* $(,)?) => {
        const _: &[fn(&mut nd_fmm_kernels::Device)] = &[$($name),*];
        #[cfg(feature = "cuda")]
        mod cuda {
            $(
                #[test]
                #[ignore = "CUDA: run by hand on locust"]
                fn $name() {
                    crate::common::run(
                        nd_fmm_kernels::BackendKind::Cuda,
                        concat!(module_path!(), "::", stringify!($name)),
                        super::$name,
                    );
                }
            )*
        }
    };
}
pub(crate) use tests_on;

/// SplitMix64: a small seeded generator for test data.
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A random permutation of `0..n`.
    pub fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            p.swap(i, self.below(i + 1));
        }
        p
    }

    /// `n` random `u32`.
    pub fn u32s(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next_u64() as u32).collect()
    }
}

/// The float types of the tests, with their bit patterns and test values.
pub trait TestFloat: DeviceFloat + std::fmt::Debug {
    /// The bits, widened.
    fn bits(self) -> u64;
    /// A value from its bits (truncated to the type's width).
    fn from_bits(bits: u64) -> Self;
    /// Rounds an f64.
    fn from_f64(x: f64) -> Self;
    /// Whether the value is a NaN.
    fn nan(self) -> bool;
    /// −0, the smallest normal, the smallest and largest subnormal, the largest finite
    /// values, ±∞, ±1, a quiet NaN and a NaN with a payload.
    fn specials() -> Vec<Self>;

    /// Any bit pattern, NaNs included.
    fn random_bits(rng: &mut Rng) -> Self {
        Self::from_bits(rng.next_u64())
    }

    /// A normal value of either sign with magnitude in [2⁻²⁰, 2²⁰): sums of two such
    /// values are never subnormal, so a flushing backend adds them exactly as the host.
    fn random_normal(rng: &mut Rng) -> Self {
        let m = 1.0 + (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let e = rng.below(40) as i32 - 20;
        let s = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        Self::from_f64(s * m * 2f64.powi(e))
    }
}

impl TestFloat for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
    fn specials() -> Vec<Self> {
        vec![
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(0x007F_FFFF),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            f32::NAN,
            f32::from_bits(0x7FC0_1234),
        ]
    }
}

impl TestFloat for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
    fn specials() -> Vec<Self> {
        vec![
            -0.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(0x000F_FFFF_FFFF_FFFF),
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            -1.0,
            f64::NAN,
            f64::from_bits(0x7FF8_0000_0000_1234),
        ]
    }
}

/// Asserts `got` equals `want` bit for bit, except that a NaN need only stay a NaN.
/// Returns whether every NaN kept its bits too.
pub fn assert_bits<T: TestFloat>(what: &str, got: &[T], want: &[T]) -> bool {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let mut payloads = true;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.nan() {
            assert!(
                g.nan(),
                "{what}: element {i}: NaN {:#x} came back as {g:?}",
                w.bits()
            );
            payloads &= g.bits() == w.bits();
        } else {
            assert_eq!(
                g.bits(),
                w.bits(),
                "{what}: element {i}: {g:?} ({:#x}) != {w:?} ({:#x})",
                g.bits(),
                w.bits()
            );
        }
    }
    payloads
}
