//! FMM3D's full tables (`tools/fetch.sh`; `FMM3D_HELMHOLTZ` names the directory).

use nd_fmm_spike_pw_quadrature::fmm3d::{self, Tables};
use std::path::Path;

#[test]
#[ignore = "needs FMM3D's sources: export FMM3D_HELMHOLTZ=$(tools/fetch.sh)"]
fn every_domain_and_precision() {
    let dir = std::env::var("FMM3D_HELMHOLTZ").expect("FMM3D_HELMHOLTZ");
    let t = Tables::load(Path::new(&dir));
    for p in 0..fmm3d::PRECISIONS {
        assert_eq!(t.precisions[p].len(), fmm3d::DOMAINS, "iprec {p}");
        for (&d, dom) in &t.precisions[p] {
            let (lo, hi) = fmm3d::rectangle(d);
            assert!(
                (dom.rk0 - lo).norm() <= 1e-14 * hi.norm()
                    && (dom.rk1 - hi).norm() <= 1e-14 * hi.norm()
            );
            // Type 2 below Im κ = π/2, type 1 above.
            assert_eq!(
                dom.kind,
                if hi.im <= core::f64::consts::FRAC_PI_2 + 1e-12 {
                    2
                } else {
                    1
                }
            );
            assert_eq!(dom.physical.len(), dom.len());
            assert_eq!(dom.fourier.len(), dom.len());
            assert!(
                dom.stated_error <= fmm3d::errmax(p),
                "{p}/{d}: {:e}",
                dom.stated_error
            );
        }
    }
}
