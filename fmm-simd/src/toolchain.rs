//! Compiled examples of the toolchain features the design relies on (crate
//! documentation, "Toolchain").
//!
//! Each example only has to compile: the module builds for its architecture in
//! `cargo test`, and for the other one in the cross-target clippy check. None calls
//! a `#[target_feature]` function from outside a matching context, which would need
//! `unsafe`; the tests take function pointers instead, which shows that such a function
//! coerces only to an `unsafe fn` pointer.

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::{
        __m256, __m256d, __m512d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_mul_ps, _mm256_rsqrt_ps,
        _mm256_set1_pd, _mm512_add_pd,
    };

    /// Rust 1.87: value-only intrinsics are safe inside a function that enables their
    /// features.
    #[target_feature(enable = "avx2,fma")]
    fn fused(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        _mm256_fmadd_pd(a, b, c)
    }

    /// Rust 1.87: the inverse-square-root estimate is one of them.
    #[target_feature(enable = "avx2,fma")]
    fn estimate(x: __m256) -> __m256 {
        let e = _mm256_rsqrt_ps(x);
        _mm256_mul_ps(e, e)
    }

    /// Rust 1.86: a safe `#[target_feature]` function calls another one with the same
    /// features without `unsafe`.
    #[target_feature(enable = "avx2,fma")]
    fn twice(a: __m256d, b: __m256d) -> __m256d {
        let x = fused(a, b, _mm256_set1_pd(1.0));
        _mm256_add_pd(x, x)
    }

    /// Rust 1.89: the AVX-512 target features and their intrinsics.
    #[target_feature(enable = "avx512f")]
    fn add512(a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[test]
    fn target_feature_functions_compile() {
        let _: unsafe fn(__m256d, __m256d) -> __m256d = twice;
        let _: unsafe fn(__m256) -> __m256 = estimate;
        let _: unsafe fn(__m512d, __m512d) -> __m512d = add512;
        println!(
            "avx512f detected: {}",
            std::arch::is_x86_feature_detected!("avx512f")
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::{
        float32x4_t, float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vmulq_f32, vrsqrteq_f32,
        vrsqrtsq_f32,
    };

    /// Rust 1.87: value-only intrinsics are safe inside a function that lists `neon`
    /// in its own `#[target_feature]`; that the target enables NEON does not suffice.
    #[target_feature(enable = "neon")]
    fn fused(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(a, b, c)
    }

    /// Rust 1.87: the estimate and the Newton-step instruction are among them.
    #[target_feature(enable = "neon")]
    fn estimate(x: float32x4_t) -> float32x4_t {
        let e = vrsqrteq_f32(x);
        vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, e), e), e)
    }

    /// Rust 1.86: a safe `#[target_feature]` function calls another one with the same
    /// features without `unsafe`.
    #[target_feature(enable = "neon")]
    fn twice(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        let x = fused(a, b, vdupq_n_f64(1.0));
        vaddq_f64(x, x)
    }

    #[test]
    fn target_feature_functions_compile() {
        let _: unsafe fn(float64x2_t, float64x2_t) -> float64x2_t = twice;
        let _: unsafe fn(float32x4_t) -> float32x4_t = estimate;
    }
}
