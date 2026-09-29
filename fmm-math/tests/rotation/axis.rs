//! Rotations about z and the z-y-z Euler angles (CONVENTIONS §3.8).

use nd_fmm_math::rotation::{block_range, blocks, blocks_len, euler_zyz};

use crate::common::SplitMix64;
use crate::support::{Matrix, mat_mul, max_abs_diff, orthonormal_block, rot_y, rot_z, square_mul};

const P: usize = 30;

/// Under a rotation by α about z, Rₙᵐ picks up the phase e^{imα} (CONVENTIONS §3.3),
/// so in real storage each pair of slots (+m, −m) turns by the angle mα:
/// [Re, Im] ↦ [cos mα, −sin mα; sin mα, cos mα] [Re, Im]. Every other entry is zero.
#[test]
fn rotation_about_z_rotates_each_order_pair() {
    let mut rng = SplitMix64::new(0x2a15_0001);
    let mut d = vec![0.0; blocks_len(P)];
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let alpha = rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        for q in [euler_zyz(alpha, 0.0, 0.0), rot_z(alpha)] {
            blocks(P, &q, &mut d);
            for n in 0..=P {
                let width = 2 * n + 1;
                let block = &d[block_range(n)];
                let at = |i: isize, j: isize| {
                    block[(i + n as isize) as usize * width + (j + n as isize) as usize]
                };
                for i in -(n as isize)..=n as isize {
                    for j in -(n as isize)..=n as isize {
                        if i.abs() != j.abs() {
                            assert_eq!(at(i, j), 0.0, "degree {n}, entry ({i}, {j})");
                        }
                    }
                }
                worst = worst.max((at(0, 0) - 1.0).abs());
                for m in 1..=n as isize {
                    let (s, c) = (m as f64 * alpha).sin_cos();
                    for (value, expected) in [
                        (at(m, m), c),
                        (at(m, -m), -s),
                        (at(-m, m), s),
                        (at(-m, -m), c),
                    ] {
                        worst = worst.max((value - expected).abs());
                    }
                }
            }
        }
    }
    println!("rotation about z: worst error {worst:.2e}");
    assert!(worst <= 1e-13, "{worst:e} > 1e-13");
}

#[test]
fn euler_zyz_composes_elementary_rotations() {
    let mut rng = SplitMix64::new(0x2a15_0002);
    for _ in 0..20 {
        let [alpha, beta, gamma] = [0; 3].map(|_| rng.range(-4.0, 4.0));
        let q: Matrix = euler_zyz(alpha, beta, gamma);
        let expected = mat_mul(&mat_mul(&rot_z(alpha), &rot_y(beta)), &rot_z(gamma));
        let err = max_abs_diff(q.as_flattened(), expected.as_flattened());
        assert!(err <= 1e-15, "euler_zyz off by {err:e}");
        // Q e_z = R_z(α) R_y(β) e_z = (sin β cos α, sin β sin α, cos β).
        let axis = [q[0][2], q[1][2], q[2][2]];
        let polar = [
            beta.sin() * alpha.cos(),
            beta.sin() * alpha.sin(),
            beta.cos(),
        ];
        assert!(max_abs_diff(&axis, &polar) <= 1e-15);
    }
}

/// D(R_z(α) R_y(β) R_z(γ)) = D(R_z(α)) D(R_y(β)) D(R_z(γ)), compared on the
/// orthonormal blocks N Dⁿ N⁻¹ as in the homomorphism test.
#[test]
fn blocks_of_euler_angles_factor() {
    let mut rng = SplitMix64::new(0x2a15_0003);
    let compute = |q: &Matrix| {
        let mut d = vec![0.0; blocks_len(P)];
        blocks(P, q, &mut d);
        d
    };
    let mut worst = 0.0_f64;
    for _ in 0..10 {
        let [alpha, beta, gamma] = [0; 3].map(|_| rng.range(-4.0, 4.0));
        let d = compute(&euler_zyz(alpha, beta, gamma));
        let factors = [rot_z(alpha), rot_y(beta), rot_z(gamma)].map(|q| compute(&q));
        for n in 0..=20 {
            let width = 2 * n + 1;
            let [a, b, c] = factors.each_ref().map(|f| orthonormal_block(f, n));
            let product = square_mul(width, &square_mul(width, &a, &b), &c);
            worst = worst.max(max_abs_diff(&orthonormal_block(&d, n), &product));
        }
    }
    println!("Euler factorisation, n <= 20: worst error {worst:.2e}");
    assert!(worst <= 1e-13, "{worst:e} > 1e-13");
}
