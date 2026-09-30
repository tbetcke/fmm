//! Single-translation accuracy report: the operator chains of `nd-fmm-ref` for the
//! standard one-box separation, against the f64 direct sum, for p = 1..=20 in f64 and
//! p = 1..=8 in f32. Prints Markdown tables on stdout.
//!
//! Run in release mode (a few minutes in debug builds would become far more):
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example accuracy
//! cargo run --release -p nd-fmm-validate --example accuracy -- --tables
//! ```
//!
//! With `--tables`, M2M, M2L and L2L are the dense tables of `nd-fmm-tables`, looked
//! up by child index and V-list offset, instead of `nd_fmm_ref::direct`. The example
//! then also runs the reference path and reports how many printed values differ.
//!
//! The geometry, chains and error measure are described in
//! `nd_fmm_validate::accuracy`. The output is deterministic for the seed.

use nd_fmm_math::RealScalar;
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_validate::accuracy::{
    Chain, Config, Errors, RADIUS, Row, SOURCE_CENTRE, Translations, sweep,
};

/// Largest degree in f64 (CONVENTIONS §3.9: M2L tested up to p = 20).
const P_MAX_F64: usize = 20;

/// Largest degree in f32 (CONVENTIONS §3.9).
const P_MAX_F32: usize = 8;

fn main() {
    let translations = match std::env::args().nth(1).as_deref() {
        None => Translations::Reference,
        Some("--tables") => Translations::Tables,
        Some(other) => {
            eprintln!("unknown argument {other:?}; the only option is --tables");
            std::process::exit(2);
        }
    };
    let config = Config::STANDARD;
    match translations {
        Translations::Reference => {
            println!("# Single-translation accuracy of the nd-fmm-ref operator chains")
        }
        Translations::Tables => {
            println!("# Single-translation accuracy of the operator chains, table path")
        }
    }
    println!();
    println!(
        "- Source box: centre {SOURCE_CENTRE:?}, half-width {RADIUS}; {} sources uniform in \
         the box, charges uniform in [-1, 1); {} targets uniform in the target box.",
        config.sources, config.targets
    );
    println!(
        "- Target box: same size, centre c + 2r·d for each of the {M2L_OFFSET_COUNT} V-list \
         offsets d."
    );
    println!(
        "- \"worst\": each error over the targets of one offset, maximised over the \
         offsets (each column separately); \"worst d\" is the offset with the largest φ L2 \
         error. \"all\": each error over the union of the targets of all offsets."
    );
    println!(
        "- Every expansion has degree p; M2M, M2L and L2L are {}. Seed {}.",
        translations.name(),
        config.seed
    );
    if translations == Translations::Tables {
        println!(
            "- Tables: `M2mTables` for M2M from source child o, `M2lTables` for M2L at \
             offset d and `L2lTables` for L2L to target child o, built in f64 at each p \
             (rounded to f32 for the f32 run). Every translation of every chain is a \
             child octant or a V-list offset, so no chain stays on `nd_fmm_ref::direct`. \
             P2M → M2P and P2L → L2P contain no translation and run on nd-fmm-ref alone."
        );
    }
    println!(
        "- Error measure: relative L2 and relative max error of the potential φ and of \
         the gradient ∇φ (Euclidean norm per target) against \
         `nd_fmm_ref::p2p::direct_sum` in f64 on the same (rounded) inputs."
    );
    let rows64 = report::<f64>("f64", &config, P_MAX_F64, translations);
    let rows32 = report::<f32>("f32", &config, P_MAX_F32, translations);
    if translations == Translations::Tables {
        println!();
        println!("## Table path against the reference path");
        println!();
        println!(
            "Every value above against the same sweep with `nd_fmm_ref::direct`, as printed \
             (three significant digits) and as computed:"
        );
        println!();
        compare::<f64>("f64", &config, P_MAX_F64, &rows64);
        compare::<f32>("f32", &config, P_MAX_F32, &rows32);
    }
}

fn report<T: RealScalar>(
    precision: &str,
    config: &Config,
    p_max: usize,
    translations: Translations,
) -> Vec<Row> {
    let rows = sweep::<T>(config, p_max, translations);
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
    rows
}

/// Runs the reference sweep and prints, for the table rows `tables`, how many printed
/// cells (the eight errors and the worst offset of each row) differ from it, and the
/// largest relative difference of the unrounded errors.
fn compare<T: RealScalar>(precision: &str, config: &Config, p_max: usize, tables: &[Row]) {
    let reference = sweep::<T>(config, p_max, Translations::Reference);
    let mut cells_total = 0;
    let mut differing = Vec::new();
    let mut largest = 0.0f64;
    for (t, r) in tables.iter().zip(&reference) {
        assert_eq!((t.chain, t.p), (r.chain, r.p));
        let (tv, rv) = (values(t), values(r));
        for (i, (a, b)) in tv.iter().zip(&rv).enumerate() {
            cells_total += 1;
            largest = largest.max((a - b).abs() / b.abs());
            if format!("{a:.2e}") != format!("{b:.2e}") {
                differing.push(format!(
                    "{}, p = {}, {}: {a:.2e} (reference {b:.2e})",
                    t.chain.name(),
                    t.p,
                    COLUMNS[i]
                ));
            }
        }
        cells_total += 1;
        if t.worst_offset != r.worst_offset {
            differing.push(format!(
                "{}, p = {}, worst d: {:?} (reference {:?})",
                t.chain.name(),
                t.p,
                t.worst_offset,
                r.worst_offset
            ));
        }
    }
    println!(
        "- {precision}: {} of {cells_total} printed cells differ; largest relative difference \
         of an error {largest:.1e}.",
        differing.len()
    );
    for d in differing {
        println!("  - {d}");
    }
}

/// The names of the eight error columns, in the order of [`values`].
const COLUMNS: [&str; 8] = [
    "worst φ L2",
    "worst φ max",
    "worst ∇φ L2",
    "worst ∇φ max",
    "all φ L2",
    "all φ max",
    "all ∇φ L2",
    "all ∇φ max",
];

/// The eight errors of a row, in the order of the printed columns.
fn values(row: &Row) -> [f64; 8] {
    let [w, a] = [&row.worst, &row.all].map(|e| {
        [
            e.potential.l2,
            e.potential.max,
            e.gradient.l2,
            e.gradient.max,
        ]
    });
    core::array::from_fn(|i| if i < 4 { w[i] } else { a[i - 4] })
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
