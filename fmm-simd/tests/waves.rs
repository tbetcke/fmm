//! Acceptance tests for the plane-wave kernels of `WaveKernel` (Phase 6 T13, C6.5;
//! CONVENTIONS §3.14) on every ISA this machine offers, in f32 and f64.
//!
//! Error measure: *bits*, exact equality of every output value by its bit pattern. Each
//! kernel equals the loop of `nd_fmm_tables::plane_wave` that defines it (`shift_add`,
//! `add` and `shift`, `waves_from_modes`, `sums_from_waves`; restated here, since this
//! crate cannot depend on nd-fmm-tables) on every ISA, so every ISA equals the scalar
//! path. The problems take wave lengths and λ-node shapes that are not multiples of any
//! lane count, so every tail runs. Every test prints the ISAs it ran.

use nd_fmm_simd::{Isa, SimdScalar, WaveBlock, WaveKernel};

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

    fn values<T: SimdScalar>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.value()).collect()
    }

    /// Below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn bits<T: SimdScalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|&v| v.to_f64().to_bits()).collect()
}

/// The ISAs this machine cannot run, for the line every test prints.
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

/// The kernels of every available ISA, after printing them under the test's name.
fn kernels<T: SimdScalar>(test: &str) -> Vec<WaveKernel<T>> {
    let kernels: Vec<WaveKernel<T>> = Isa::available()
        .map(|isa| WaveKernel::new(isa).expect("an available ISA"))
        .collect();
    let names: Vec<String> = kernels.iter().map(|k| k.isa().to_string()).collect();
    println!(
        "{test} ({}): ISAs run: {}; not available here: {}",
        std::any::type_name::<T>(),
        names.join(", "),
        unavailable()
    );
    kernels
}

/// `shift_add` of nd-fmm-tables, source by source.
fn naive_merge<T: SimdScalar>(
    acc: &mut [T],
    waves: &[T],
    wave_offsets: &[usize],
    shifts: &[T],
    shift_offsets: &[usize],
) {
    let h = acc.len() / 2;
    for (&wo, &so) in wave_offsets.iter().zip(shift_offsets) {
        for i in 0..h {
            let (xr, xi) = (waves[wo + i], waves[wo + h + i]);
            let (tr, ti) = (shifts[so + i], shifts[so + h + i]);
            acc[i] = xr.mul_add(tr, acc[i]);
            acc[i] = (-xi).mul_add(ti, acc[i]);
            acc[h + i] = xr.mul_add(ti, acc[h + i]);
            acc[h + i] = xi.mul_add(tr, acc[h + i]);
        }
    }
}

/// `add` from zero, then `shift`, of nd-fmm-tables.
fn naive_gather<T: SimdScalar>(v: &mut [T], accs: &[T], acc_offsets: &[usize], t: &[T]) {
    v.fill(T::zero());
    for &o in acc_offsets {
        for (v, &a) in v.iter_mut().zip(&accs[o..]) {
            *v = *v + a;
        }
    }
    let h = v.len() / 2;
    for i in 0..h {
        let (a, b) = (v[i], v[h + i]);
        v[i] = (-b).mul_add(t[h + i], a * t[i]);
        v[h + i] = a.mul_add(t[h + i], b * t[i]);
    }
}

/// λ nodes of random shapes and the tables of their cos and sin rows.
struct Layout<T> {
    blocks: Vec<WaveBlock>,
    h: usize,
    modes: usize,
    cos: Vec<T>,
    sin: Vec<T>,
}

impl<T: SimdScalar> Layout<T> {
    fn new(nodes: usize, max_modes: usize, rng: &mut Rng) -> Self {
        let (mut blocks, mut start, mut mode_start, mut trig) = (Vec::new(), 0, 0, 0);
        for _ in 0..nodes {
            let (len, modes) = (1 + rng.below(31), 1 + rng.below(max_modes));
            blocks.push(WaveBlock {
                start,
                len,
                modes,
                mode_start,
                trig,
            });
            start += len;
            mode_start += modes;
            trig += len * modes;
        }
        let (cos, sin) = (rng.values(trig), rng.values(trig));
        Self {
            blocks,
            h: start,
            modes: mode_start,
            cos,
            sin,
        }
    }

    /// `waves_from_modes` of nd-fmm-tables.
    fn naive_waves(&self, modes: &[T], up: &mut [T], down: &mut [T]) {
        up.fill(T::zero());
        down.fill(T::zero());
        for b in &self.blocks {
            for m in 0..b.modes {
                let k = 4 * (b.mode_start + m);
                let half = if m % 2 == 0 { 0 } else { self.h };
                for (wave, (a, bb)) in [
                    (&mut *up, (modes[k], modes[k + 1])),
                    (&mut *down, (modes[k + 2], modes[k + 3])),
                ] {
                    for j in 0..b.len {
                        let x = &mut wave[half + b.start + j];
                        *x = a.mul_add(self.cos[b.trig + m * b.len + j], *x);
                        *x = bb.mul_add(self.sin[b.trig + m * b.len + j], *x);
                    }
                }
            }
        }
    }

    /// `sums_from_waves` of nd-fmm-tables.
    fn naive_sums(&self, up: &[T], down: &[T], sums: &mut [T]) {
        for b in &self.blocks {
            for m in 0..b.modes {
                let half = if m % 2 == 0 { 0 } else { self.h };
                for (q, wave) in [up, down].into_iter().enumerate() {
                    let (mut sc, mut ss) = (T::zero(), T::zero());
                    for j in 0..b.len {
                        let x = wave[half + b.start + j];
                        sc = x.mul_add(self.cos[b.trig + m * b.len + j], sc);
                        ss = x.mul_add(self.sin[b.trig + m * b.len + j], ss);
                    }
                    let k = 4 * (b.mode_start + m) + 2 * q;
                    sums[k] = sc;
                    sums[k + 1] = ss;
                }
            }
        }
    }
}

fn merge_and_gather<T: SimdScalar>() {
    let kernels = kernels::<T>("merge and gather");
    for (case, &h) in [0, 1, 3, 7, 8, 15, 16, 17, 33, 100, 281].iter().enumerate() {
        let mut rng = Rng(0x3e7 + case as u64);
        let len = 2 * h;
        let sources = 1 + rng.below(9);
        let waves = rng.values::<T>(len * 5 + 3);
        let shifts = rng.values::<T>(len * 4 + 1);
        let wave_offsets: Vec<usize> = (0..sources).map(|_| rng.below(4 * len + 4)).collect();
        let shift_offsets: Vec<usize> = (0..sources).map(|_| rng.below(3 * len + 2)).collect();
        let start = rng.values::<T>(len);
        let mut expected = start.clone();
        naive_merge(
            &mut expected,
            &waves,
            &wave_offsets,
            &shifts,
            &shift_offsets,
        );
        let accs = rng.values::<T>(len * 6);
        let acc_offsets: Vec<usize> = (0..rng.below(6)).map(|_| rng.below(5 * len + 1)).collect();
        let t = rng.values::<T>(len);
        let mut gathered = vec![T::zero(); len];
        naive_gather(&mut gathered, &accs, &acc_offsets, &t);
        for kernel in &kernels {
            let mut acc = start.clone();
            kernel.merge(&mut acc, &waves, &wave_offsets, &shifts, &shift_offsets);
            assert_eq!(
                bits(&acc),
                bits(&expected),
                "merge, H = {h}, {}",
                kernel.isa()
            );
            let mut v = rng.values::<T>(len);
            kernel.gather(&mut v, &accs, &acc_offsets, &t);
            assert_eq!(
                bits(&v),
                bits(&gathered),
                "gather, H = {h}, {}",
                kernel.isa()
            );
        }
    }
}

#[test]
fn merge_and_gather_equal_the_reference_loops_on_every_isa() {
    merge_and_gather::<f32>();
    merge_and_gather::<f64>();
}

fn fourier_passes<T: SimdScalar>() {
    let kernels = kernels::<T>("waves and sums");
    for (case, (nodes, max_modes)) in [(1, 1), (3, 4), (18, 9), (32, 19), (7, 21)]
        .into_iter()
        .enumerate()
    {
        let mut rng = Rng(0xf0 + case as u64);
        let layout = Layout::<T>::new(nodes, max_modes, &mut rng);
        let modes = rng.values::<T>(4 * layout.modes);
        let (mut up, mut down) = (vec![T::zero(); 2 * layout.h], vec![T::zero(); 2 * layout.h]);
        layout.naive_waves(&modes, &mut up, &mut down);
        let (in_up, in_down) = (rng.values::<T>(2 * layout.h), rng.values::<T>(2 * layout.h));
        let mut sums = vec![T::zero(); 4 * layout.modes];
        layout.naive_sums(&in_up, &in_down, &mut sums);
        for kernel in &kernels {
            let tables = kernel.tables(&layout.blocks, &layout.cos, &layout.sin);
            let (mut u, mut d) = (rng.values::<T>(2 * layout.h), rng.values::<T>(2 * layout.h));
            kernel.waves(&tables, &modes, &mut u, &mut d);
            assert_eq!(
                bits(&u),
                bits(&up),
                "waves (up), case {case}, {}",
                kernel.isa()
            );
            assert_eq!(
                bits(&d),
                bits(&down),
                "waves (down), case {case}, {}",
                kernel.isa()
            );
            let mut s = rng.values::<T>(4 * layout.modes);
            kernel.sums(&tables, &in_up, &in_down, &mut s);
            assert_eq!(bits(&s), bits(&sums), "sums, case {case}, {}", kernel.isa());
        }
    }
}

#[test]
fn fourier_passes_equal_the_reference_loops_on_every_isa() {
    fourier_passes::<f32>();
    fourier_passes::<f64>();
}

#[test]
#[should_panic(expected = "another kernel's lanes")]
fn tables_belong_to_their_kernel() {
    let mut rng = Rng(7);
    let layout = Layout::<f64>::new(3, 4, &mut rng);
    let tables = WaveKernel::<f64>::new(Isa::Scalar).unwrap().tables(
        &layout.blocks,
        &layout.cos,
        &layout.sin,
    );
    let wide = Isa::available().find(|isa| *isa != Isa::Scalar);
    let (mut up, mut down) = (vec![0.0; 2 * layout.h], vec![0.0; 2 * layout.h]);
    let modes = vec![0.0; 4 * layout.modes];
    match wide {
        Some(isa) => WaveKernel::<f64>::new(isa)
            .unwrap()
            .waves(&tables, &modes, &mut up, &mut down),
        None => panic!("the tables were made for another kernel's lanes (no vector ISA here)"),
    }
}

#[test]
#[should_panic(expected = "ends beyond")]
fn merge_checks_its_offsets() {
    let kernel = WaveKernel::<f64>::detect();
    let mut acc = [0.0; 4];
    kernel.merge(&mut acc, &[0.0; 6], &[3], &[0.0; 8], &[0]);
}
