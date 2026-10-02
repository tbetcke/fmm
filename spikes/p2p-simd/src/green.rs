//! green-kernels' Laplace kernel at the pinned commit, single-threaded, in its own
//! layout: interleaved triples in, one value (`Value`) or four interleaved values
//! (`ValueDeriv`: φ, ∂x, ∂y, ∂z) per target out, scaled by 1/(4π).

use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel;
use green_kernels::types::GreenKernelEvalType;

use crate::simd::Elem;

/// The precisions green-kernels is run in.
pub trait Gk: Elem {
    /// `Laplace3dKernel::<Self>::evaluate_st`, adding into `out` (n_t or 4 n_t values).
    fn gk(
        grad: bool,
        sources: &[[Self; 3]],
        charges: &[Self],
        targets: &[[Self; 3]],
        out: &mut [Self],
    );
}

fn eval_type(grad: bool) -> GreenKernelEvalType {
    if grad {
        GreenKernelEvalType::ValueDeriv
    } else {
        GreenKernelEvalType::Value
    }
}

impl Gk for f32 {
    fn gk(grad: bool, s: &[[f32; 3]], q: &[f32], t: &[[f32; 3]], out: &mut [f32]) {
        Laplace3dKernel::<f32>::new().evaluate_st(
            eval_type(grad),
            s.as_flattened(),
            t.as_flattened(),
            q,
            out,
        );
    }
}

impl Gk for f64 {
    fn gk(grad: bool, s: &[[f64; 3]], q: &[f64], t: &[[f64; 3]], out: &mut [f64]) {
        Laplace3dKernel::<f64>::new().evaluate_st(
            eval_type(grad),
            s.as_flattened(),
            t.as_flattened(),
            q,
            out,
        );
    }
}

/// green-kernels' output in our form, in f64, with its 1/(4π) removed: potentials and
/// optional gradients.
pub fn unpack<E: Elem>(grad: bool, out: &[E], n: usize) -> (Vec<f64>, Option<Vec<[f64; 3]>>) {
    let s = 4.0 * std::f64::consts::PI;
    if grad {
        let pot = (0..n).map(|i| out[4 * i].to_f64() * s).collect();
        let g = (0..n)
            .map(|i| [1, 2, 3].map(|c| out[4 * i + c].to_f64() * s))
            .collect();
        (pot, Some(g))
    } else {
        ((0..n).map(|i| out[i].to_f64() * s).collect(), None)
    }
}
