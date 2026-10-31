//! The Helmholtz leaf rule over decision D1's full range (Helmholtz CH3.0b; proposal
//! §3.7, §9.1): W1 and W2 at kD = 10 to 500 (W2 3.2 million points at 500), W3 resolved
//! at kD = 10, 30 and 100 (4.0 million points) and under-resolved at N = 10⁶ for kD = 200
//! and 500, and W4's four distributions at N = 10⁵ for kD = 1 to 500. No leaf that holds
//! a point is wider than c_leaf λ, and the table of the leaves is printed for the record
//! (docs/helmholtz/tasks/CH3.0b.md, "Results").
//!
//! Ignored: 7 s in release on the M6, far longer in debug (W3 resolved at kD = 100 has
//! 4.0 million points). Run by hand:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate --release \
//!     --test helmholtz_leaves_sweep -- --ignored --nocapture
//! ```

mod helmholtz_leaves_common;

use std::time::Instant;

use helmholtz_leaves_common::{check, fine_keys};
use nd_fmm_exec::helmholtz::Wavenumber;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_workloads::Workload;

#[test]
#[ignore = "the full kD range: millions of points, run by hand in release"]
fn the_rule_over_the_full_kd_range() {
    let _universe = mpi::initialize().expect("MPI initialises once");

    let mut cases: Vec<(Workload, f64)> = Vec::new();
    for workload in [Workload::W1, Workload::W2] {
        for kd in [10.0, 30.0, 100.0, 200.0, 300.0, 500.0] {
            cases.push((workload, kd));
        }
    }
    for kd in [10.0, 30.0, 100.0] {
        cases.push((Workload::w3_resolved(kd), kd));
    }
    for kd in [200.0, 500.0] {
        cases.push((Workload::W3 { n: 1_000_000 }, kd));
    }
    for distribution in [
        Distribution::Cube,
        Distribution::Sphere,
        Distribution::Plummer,
        Distribution::Clusters,
    ] {
        for kd in [1.0, 10.0, 100.0, 500.0] {
            cases.push((
                Workload::W4 {
                    distribution,
                    n: 100_000,
                },
                kd,
            ));
        }
    }

    println!(
        "| Workload | kD | N | rule level | leaves | occupied | occupied levels | widest (λ) | seconds |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for (workload, kd) in cases {
        let start = Instant::now();
        let name = workload.name();
        let points = workload.points(kd);
        let keys = fine_keys(&points);
        let k = Wavenumber::new(kd, 0.0).unwrap();
        let leaves = check(&name, &keys, &k);
        println!(
            "| {name} | {kd} | {} | {} | {} | {} | {}–{} | {:.3} | {:.1} |",
            points.len(),
            leaves.rule,
            leaves.leaves,
            leaves.occupied,
            leaves.levels.0,
            leaves.levels.1,
            leaves.widest.unwrap(),
            start.elapsed().as_secs_f64()
        );
    }
}
