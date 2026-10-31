//! The Helmholtz leaf rule on the reference workloads (Helmholtz CH3.0b; proposal §3.7,
//! §9 item 8): no leaf that holds a point is wider than c_leaf λ, on the trees of W1–W4
//! at kD within decision D1's range (kD ≤ 500), with the debug-sized subset here. Complex
//! k has the rule of real Re k, so its tree; Yukawa adds no rule, its tree `Fmm`'s tree
//! key for key. The full D1 range is `tests/helmholtz_leaves_sweep.rs` (ignored, release).
//!
//! One test initialises MPI; each rank checks whole trees on its own communicator
//! (`helmholtz_leaves_common`), so it passes on any number of ranks.

mod helmholtz_leaves_common;

use std::time::Instant;

use helmholtz_leaves_common::{check, domain, fine_keys, leaf_keys, leaf_keys_without_rule};
use nd_fmm_exec::helmholtz::{C_LEAF, Wavenumber, min_leaf_level};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_workloads::Workload;

const DISTRIBUTIONS: [Distribution; 4] = [
    Distribution::Cube,
    Distribution::Sphere,
    Distribution::Plummer,
    Distribution::Clusters,
];

#[test]
fn no_occupied_leaf_wider_than_c_leaf_lambda() {
    let _universe = mpi::initialize().expect("MPI initialises once");

    let mut cases: Vec<(Workload, f64)> = Vec::new();
    for kd in [10.0, 30.0, 100.0] {
        cases.push((Workload::W1, kd));
        cases.push((Workload::W2, kd));
    }
    for kd in [10.0, 100.0, 500.0] {
        cases.push((Workload::W3 { n: 20_000 }, kd));
    }
    for distribution in DISTRIBUTIONS {
        for kd in [1.0, 10.0, 100.0, 500.0] {
            cases.push((
                Workload::W4 {
                    distribution,
                    n: 10_000,
                },
                kd,
            ));
        }
    }

    let mut yukawa_checked: Vec<Workload> = Vec::new();
    for (workload, kd) in cases {
        let start = Instant::now();
        let name = workload.name();
        let keys = fine_keys(&workload.points(kd));
        let real = Wavenumber::new(kd, 0.0).unwrap();
        let leaves = check(&name, &keys, &real);

        // Complex k: the rule of real Re k, so the same options and the same tree.
        let complex = Wavenumber::new(kd, 0.1 * kd).unwrap();
        assert_eq!(
            min_leaf_level(&complex, &domain(), C_LEAF),
            leaves.rule,
            "{name}, kD = {kd}: complex k"
        );

        // Yukawa: no rule, `Fmm`'s tree key for key; once per point set.
        let yukawa = Wavenumber::new(0.0, kd).unwrap();
        assert_eq!(min_leaf_level(&yukawa, &domain(), C_LEAF), 0);
        let depends_on_kd = matches!(workload, Workload::W1 | Workload::W2);
        if depends_on_kd || !yukawa_checked.contains(&workload) {
            assert_eq!(
                leaf_keys(&keys, &yukawa),
                leaf_keys_without_rule(&keys),
                "{name}, κD = {kd}: Yukawa"
            );
            yukawa_checked.push(workload);
        }
        println!(
            "{name}, kD = {kd}: rule level {}, {} of {} leaves occupied on levels {}–{}, \
             widest {:.3} λ ({:.1} s)",
            leaves.rule,
            leaves.occupied,
            leaves.leaves,
            leaves.levels.0,
            leaves.levels.1,
            leaves.widest.unwrap(),
            start.elapsed().as_secs_f64()
        );
    }
}
