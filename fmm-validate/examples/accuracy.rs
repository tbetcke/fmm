//! Single-translation accuracy report: the operator chains of `nd-fmm-ref` for the
//! standard one-box separation, against the f64 direct sum, for p = 1..=20 in f64 and
//! p = 1..=8 in f32. Prints Markdown tables on stdout.
//!
//! Run in release mode (a few minutes in debug builds would become far more):
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example accuracy
//! ```
//!
//! The geometry, chains and error measure are described in
//! `nd_fmm_validate::accuracy`. The output is deterministic for the seed.

use nd_fmm_math::RealScalar;
use nd_fmm_validate::accuracy::{
    Chain, Config, Errors, RADIUS, Row, SOURCE_CENTRE, sweep, v_list_offsets,
};

/// Largest degree in f64 (CONVENTIONS §3.9: M2L tested up to p = 20).
const P_MAX_F64: usize = 20;

/// Largest degree in f32 (CONVENTIONS §3.9).
const P_MAX_F32: usize = 8;

fn main() {
    let config = Config::STANDARD;
    println!("# Single-translation accuracy of the nd-fmm-ref operator chains");
    println!();
    println!(
        "- Source box: centre {SOURCE_CENTRE:?}, half-width {RADIUS}; {} sources uniform in \
         the box, charges uniform in [-1, 1); {} targets uniform in the target box.",
        config.sources, config.targets
    );
    println!(
        "- Target box: same size, centre c + 2r·d for each of the {} V-list offsets d.",
        v_list_offsets().len()
    );
    println!(
        "- \"worst\": each error over the targets of one offset, maximised over the \
         offsets (each column separately); \"worst d\" is the offset with the largest φ L2 \
         error. \"all\": each error over the union of the targets of all offsets."
    );
    println!(
        "- Every expansion has degree p; M2M, M2L and L2L are `nd_fmm_ref::direct`. \
         Seed {}.",
        config.seed
    );
    println!(
        "- Error measure: relative L2 and relative max error of the potential φ and of \
         the gradient ∇φ (Euclidean norm per target) against \
         `nd_fmm_ref::p2p::direct_sum` in f64 on the same (rounded) inputs."
    );
    report::<f64>("f64", &config, P_MAX_F64);
    report::<f32>("f32", &config, P_MAX_F32);
}

fn report<T: RealScalar>(precision: &str, config: &Config, p_max: usize) {
    let rows = sweep::<T>(config, p_max);
    println!();
    println!("## {precision}, p = 1..={p_max}");
    for chain in Chain::ALL {
        println!();
        println!("### {} ({precision})", chain.name());
        println!();
        println!(
            "| p | worst d | worst φ L2 | worst φ max | worst ∇φ L2 | worst ∇φ max \
             | all φ L2 | all φ max | all ∇φ L2 | all ∇φ max |"
        );
        println!("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for row in rows.iter().filter(|r: &&Row| r.chain == chain) {
            println!(
                "| {} | {:?} | {} | {} |",
                row.p,
                row.worst_offset,
                cells(&row.worst),
                cells(&row.all)
            );
        }
    }
}

fn cells(e: &Errors) -> String {
    [
        e.potential.l2,
        e.potential.max,
        e.gradient.l2,
        e.gradient.max,
    ]
    .map(|v| format!("{v:.2e}"))
    .join(" | ")
}
