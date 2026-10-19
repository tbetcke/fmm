//! The plane-wave kernels bound to one ISA: the shifts, the accumulations and the Fourier
//! passes of the host's plane-wave M2L (Phase 6 T13, C6.5; CONVENTIONS §3.14).

use std::marker::PhantomData;

use crate::arch::AvailableIsa;
use crate::{Isa, IsaUnavailable, SimdScalar};

/// One λ node of plane waves: where its values lie in the waves and the modes, and its
/// rows in the cos and sin tables (`nd_fmm_tables::plane_wave::LambdaBlock`, field for
/// field), from which [`WaveKernel::tables`] builds the kernel's layout.
///
/// A direction's waves hold 2H values, the H real parts and then the H imaginary parts;
/// the node's values are `start .. start + len` of each half.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WaveBlock {
    /// Its first value among the H values of a half.
    pub start: usize,
    /// Its values.
    pub len: usize,
    /// Its Fourier modes F: orders 0 ≤ m < F.
    pub modes: usize,
    /// Its first mode among the modes of every node: the modes and the sums hold four
    /// values per mode, at 4 (`mode_start` + m).
    pub mode_start: usize,
    /// Its first value in the cos and sin tables: F rows of `len` values, row m at
    /// `trig + m len`, cos mαⱼ and sin mαⱼ.
    pub trig: usize,
}

pub use crate::arch::waves::WaveTables;

/// The plane-wave kernels bound to one available [`Isa`]: the loops of the host's
/// plane-wave M2L over the values of the waves (Phase 6 T13; `nd_fmm_tables::plane_wave`
/// defines each).
///
/// | Method | Computes | Chain per value (each fma rounded once) |
/// | --- | --- | --- |
/// | [`merge`](Self::merge) | acc += Σᵢ xᵢ ⊙ tᵢ | per source in order: fma(x_r, t_r), fnma(x_i, t_i), fma(x_r, t_i), fma(x_i, t_r), as `shift_add` |
/// | [`gather`](Self::gather) | v ← (Σᵢ aᵢ) ⊙ t | from zero, aᵢ added in order (`add`), then `shift` |
/// | [`waves`](Self::waves) | the waves from the weighted modes | per mode ascending fma(a, cos), fma(b, sin), from zero (`waves_from_modes`) |
/// | [`sums`](Self::sums) | the incoming sums Σⱼ x cos, Σⱼ x sin | per row j ascending one fma, from zero (`sums_from_waves`) |
///
/// Every value depends on its own inputs only and receives the same chain on every ISA,
/// so the results do not depend on the ISA, bit for bit (IEEE 754 fma, multiplication and
/// addition are correctly rounded everywhere; the scalar path forms each fma with
/// `mul_add`). Complex waves hold 2H values: the H real parts, then the H imaginary parts.
/// The kernel is stateless: `Copy`, `Send` and `Sync`.
///
/// ```
/// use nd_fmm_simd::{Isa, WaveKernel};
///
/// let kernel = WaveKernel::<f64>::new(Isa::detect()).unwrap();
/// // Two complex values: x = (1 + 2i, 3 + 0i), t = (0 + 1i, 2 + 0i), acc = 0.
/// let x = [1.0, 3.0, 2.0, 0.0];
/// let t = [0.0, 2.0, 1.0, 0.0];
/// let mut acc = [0.0; 4];
/// kernel.merge(&mut acc, &x, &[0], &t, &[0]);
/// assert_eq!(acc, [-2.0, 6.0, 1.0, 0.0]); // (1 + 2i) i = −2 + i, 3 · 2 = 6
/// ```
#[derive(Clone, Copy, Debug)]
pub struct WaveKernel<T: SimdScalar> {
    /// The ISA, checked to be available by every constructor.
    isa: AvailableIsa,
    precision: PhantomData<T>,
}

impl<T: SimdScalar> WaveKernel<T> {
    /// The kernels on `isa`.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] if this machine cannot run `isa` ([`Isa::is_available`]).
    pub fn new(isa: Isa) -> Result<Self, IsaUnavailable> {
        Ok(Self {
            isa: AvailableIsa::new(isa)?,
            precision: PhantomData,
        })
    }

    /// The kernels on the default ISA of this machine, [`Isa::detect`].
    pub fn detect() -> Self {
        Self::new(Isa::detect()).expect("`Isa::detect` returns an available ISA")
    }

    /// The ISA the kernels run on.
    pub fn isa(&self) -> Isa {
        self.isa.isa()
    }

    /// acc += Σᵢ xᵢ ⊙ tᵢ, complex, over waves of 2H = `acc.len()` values: xᵢ the 2H values
    /// of `waves` at `wave_offsets[i]`, tᵢ those of `shifts` at `shift_offsets[i]`, the
    /// sources in order (table of [`WaveKernel`]). Merge and shift's accumulation of one
    /// parent's sources: the accumulator stays in registers across the sources.
    ///
    /// # Panics
    ///
    /// If `acc` has an odd length, the offset lists differ in length, or a wave or shift
    /// lies outside its slice.
    pub fn merge(
        &self,
        acc: &mut [T],
        waves: &[T],
        wave_offsets: &[usize],
        shifts: &[T],
        shift_offsets: &[usize],
    ) {
        let len = acc.len();
        assert!(len.is_multiple_of(2), "a wave holds 2H values, not {len}");
        assert_eq!(
            wave_offsets.len(),
            shift_offsets.len(),
            "one shift per wave"
        );
        check_offsets("waves", waves.len(), wave_offsets, len);
        check_offsets("shifts", shifts.len(), shift_offsets, len);
        if wave_offsets.is_empty() || len == 0 {
            return;
        }
        T::merge(self.isa, acc, waves, wave_offsets, shifts, shift_offsets);
    }

    /// v ← (Σᵢ aᵢ) ⊙ t, complex, over waves of 2H = `v.len()` values: aᵢ the 2H values of
    /// `accs` at `acc_offsets[i]`, added in order from zero, then the product with `t`
    /// (table of [`WaveKernel`]). Merge and shift's incoming waves of one child: its
    /// accumulators and its shift. With no accumulator, v is zero.
    ///
    /// # Panics
    ///
    /// If `v` has an odd length, `t` another length, or an accumulator lies outside `accs`.
    pub fn gather(&self, v: &mut [T], accs: &[T], acc_offsets: &[usize], t: &[T]) {
        let len = v.len();
        assert!(len.is_multiple_of(2), "a wave holds 2H values, not {len}");
        assert_eq!(t.len(), len, "the shift holds 2H values");
        check_offsets("accs", accs.len(), acc_offsets, len);
        if len == 0 {
            return;
        }
        T::gather(self.isa, v, accs, acc_offsets, t);
    }

    /// The tables of [`waves`](Self::waves) and [`sums`](Self::sums) in this kernel's
    /// layout ([`WaveTables`]): the λ nodes `blocks` with the rows of `cos` and `sin`,
    /// padded to the kernel's lanes ([`Isa::lanes`]) with zeros, and the incoming pairs
    /// (cos mαⱼ, sin mαⱼ) by value, the even modes' and then the odd modes', each padded
    /// likewise. Allocates; build once.
    ///
    /// # Panics
    ///
    /// If a node's rows lie outside `cos` or `sin`, the tables differ in length, or a node
    /// has more than 63 modes.
    pub fn tables(&self, blocks: &[WaveBlock], cos: &[T], sin: &[T]) -> WaveTables<T> {
        assert_eq!(cos.len(), sin.len(), "cos and sin tables of one length");
        for block in blocks {
            assert!(block.modes < 64, "at most 63 modes per λ node");
            assert!(
                block.trig + block.modes * block.len <= cos.len(),
                "the rows of a λ node lie outside the tables"
            );
        }
        WaveTables::new(blocks, cos, sin, self.isa().lanes::<T>())
    }

    /// The up and down waves (2H values each) of the weighted modes `modes`, four per mode
    /// (a↑, b↑, a↓, b↓): for each λ node of `tables` and value j, from zero, for every mode
    /// m ascending x ← fma(a, cos mαⱼ, x), x ← fma(b, sin mαⱼ, x) into the real parts for
    /// even m and the imaginary parts for odd m. Writes the values of the nodes.
    ///
    /// # Panics
    ///
    /// If `tables` was made for another lane count ([`tables`](Self::tables)), `up` and
    /// `down` differ in length or have an odd one, or a node lies outside the waves or
    /// `modes`.
    pub fn waves(&self, tables: &WaveTables<T>, modes: &[T], up: &mut [T], down: &mut [T]) {
        self.check_tables(tables, modes.len(), up.len(), down.len());
        T::waves(self.isa, tables, modes, up, down);
    }

    /// The incoming sums of the up and down waves (2H values each), four per mode:
    /// (Σⱼ x↑ cos mαⱼ, Σⱼ x↑ sin mαⱼ, Σⱼ x↓ cos mαⱼ, Σⱼ x↓ sin mαⱼ) with x the real parts for
    /// even m and the imaginary parts for odd m, each from zero over j ascending, one fma
    /// per term. Writes the sums of the nodes' modes.
    ///
    /// # Panics
    ///
    /// As [`waves`](Self::waves), with `sums` for `modes`.
    pub fn sums(&self, tables: &WaveTables<T>, up: &[T], down: &[T], sums: &mut [T]) {
        self.check_tables(tables, sums.len(), up.len(), down.len());
        T::sums(self.isa, tables, up, down, sums);
    }

    /// Panics unless `tables` has this kernel's lanes and its nodes fit waves of `up`
    /// (= `down`) values and `modes` values of four per mode.
    fn check_tables(&self, tables: &WaveTables<T>, modes: usize, up: usize, down: usize) {
        assert_eq!(
            tables.lanes,
            self.isa().lanes::<T>(),
            "the tables were made for another kernel's lanes"
        );
        assert!(
            up == down && up.is_multiple_of(2),
            "up and down waves of one even length, not {up} and {down}"
        );
        assert!(tables.h <= up / 2, "a λ node lies outside the waves");
        assert!(
            4 * tables.modes <= modes,
            "a λ node's modes lie outside the four values per mode"
        );
    }
}

/// Panics unless every offset + `len` lies within `values`.
fn check_offsets(name: &str, values: usize, offsets: &[usize], len: usize) {
    for &offset in offsets {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= values),
            "the wave of `{name}` at {offset} ends beyond its {values} values"
        );
    }
}
