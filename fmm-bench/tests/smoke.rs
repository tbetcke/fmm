//! The smoke run of the benchmark (Phase 4S T6): the host at N = 2,000, p = 3 in f32 and
//! f64 with two repeats, and with `--features cpu` the same on the CubeCL CPU runtime,
//! into a temporary file, read back and checked: a summary row per combination, every
//! one measured, and a kind table whose kinds are non-negative and sum to at most the
//! mean evaluation of the kind-timed evaluations, with the charge load, the output pass
//! and the rest ("load", "output", "other"; Phase 4S T9) inside it. It runs twice: with the
//! default `Fmm::evaluate` and with `--reuse-output` (`Fmm::evaluate_into`), each report's
//! header naming the call. Timings are read, never asserted beyond that.
//!
//! The kind times of one evaluation are level calls inside its stages, and the stages lie
//! inside the wall time of `Fmm::evaluate`, so the sum of the kinds is at most that
//! evaluation's wall time in every mode that times on the host; the bound is checked
//! against the kind-timed evaluations' own mean ("evaluation" column), not the summary's,
//! which comes from other evaluations.
//!
//! This executable initialises MPI once, in its one test.

use mpi::Threading;
use nd_fmm_bench::measure::{SEED, run};
use nd_fmm_bench::options::{Command, parse};
use nd_fmm_bench::report::{Header, KIND_HEADER, Report, SUMMARY_HEADER};
use nd_fmm_exec::fmm::Backend;

/// The cells of the rows of the Markdown table whose header (and alignment row) is
/// `header`.
fn table(text: &str, header: &str) -> Vec<Vec<String>> {
    let start = text.find(header).expect("the table is in the report") + header.len();
    text[start..]
        .lines()
        .skip(1)
        .take_while(|line| line.starts_with('|'))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_owned())
                .collect()
        })
        .collect()
}

#[test]
fn smoke_run_writes_a_row_and_kinds_per_combination() {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();

    let mut backends = vec![Backend::Host];
    if Backend::Cpu.is_compiled() {
        backends.push(Backend::Cpu);
    }
    let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
    let backend_list = names.join(",");
    // The default, `Fmm::evaluate`, then `--reuse-output`, `Fmm::evaluate_into` (Phase 4S
    // T9): the same structure, the header naming which ran.
    let mut text = String::new();
    for reuse in [false, true] {
        let output = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "nd-fmm-bench-smoke-{}-{reuse}.md",
            std::process::id()
        ));
        let mut args = vec![
            "--n",
            "2000",
            "--precision",
            "f32,f64",
            "--degree",
            "3",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--threads",
            "2",
            "--accuracy",
            "200",
            "--backend",
            &backend_list,
        ];
        if reuse {
            args.push("--reuse-output");
        }
        let Ok(Command::Run(options)) = parse(&args) else {
            panic!("the smoke arguments parse");
        };
        let mut report = Report {
            header: Header::collect(&options, format!("smoke {}", args.join(" "))),
            options: *options,
            seed: SEED,
            rows: Vec::new(),
        };
        run(&mut report, &output, &comm).expect("the report is written");
        text = std::fs::read_to_string(&output).expect("the report is read back");
        std::fs::remove_file(&output).expect("the report is removed");

        let summary = table(&text, SUMMARY_HEADER);
        let kinds = table(&text, KIND_HEADER);
        let combinations = 2 * backends.len();
        assert_eq!(
            summary.len(),
            combinations,
            "one summary row per combination"
        );
        assert_eq!(kinds.len(), combinations, "one kind row per combination");
        let number = |cell: &str| -> f64 {
            cell.parse()
                .unwrap_or_else(|_| panic!("`{cell}` is a number"))
        };
        for (row, kind) in summary.iter().zip(&kinds) {
            assert_eq!(row[..4], kind[..4], "the same combination in both tables");
            assert!(
                !row[4].starts_with("refused") && !row[4].starts_with("failed"),
                "{row:?} ran"
            );
            assert_eq!(row[13], "yes", "{row:?}: the repeats bit for bit");
            // min ≤ median ≤ max and min ≤ mean ≤ max, in ms.
            let (min, median, mean, max) = (
                number(&row[6]),
                number(&row[7]),
                number(&row[8]),
                number(&row[9]),
            );
            assert!(min > 0.0 && min <= median && median <= max && min <= mean && mean <= max);
            // A sanity bound on the errors: p = 3 at the 1e-3 level.
            assert!(
                number(&row[11]) < 1e-2 && number(&row[12]) < 1e-1,
                "{row:?}"
            );
            // The kinds: P2M … P2P, load, output, other, sum, evaluation, sum / mean.
            let values: Vec<f64> = kind[4..18].iter().map(|c| number(c)).collect();
            assert!(values.iter().all(|&v| v >= 0.0), "{kind:?}");
            let sum: f64 = values[..8].iter().sum();
            let (listed, evaluation) = (number(&kind[15]), number(&kind[16]));
            // Load, output and other lie outside the level calls, inside the evaluation.
            assert!(
                sum + values[8..11].iter().sum::<f64>() <= evaluation + 0.004,
                "{kind:?}: the kinds, load, output and other within the mean evaluation"
            );
            assert!(values[9] > 0.0, "{kind:?}: the output pass is timed");
            // The cells are rounded to 0.001 ms.
            assert!(
                (sum - listed).abs() <= 0.005,
                "{kind:?}: the kinds add up to the sum"
            );
            assert!(
                listed <= evaluation + 0.001,
                "{kind:?}: the kinds sum to at most the mean evaluation"
            );
            assert!(
                values[2] > 0.0 && values[7] > 0.0,
                "{kind:?}: M2L and P2P ran"
            );
        }
        assert!(
            text.contains(if reuse {
                "| output | one `Output` reused by every evaluation"
            } else {
                "| output | a fresh `Output` per evaluation"
            }),
            "the header names the evaluation call"
        );
        // The download's part of the output, on each device row.
        assert_eq!(
            text.matches("Output: of it the download").count(),
            2 * (backends.len() - 1),
            "one download line per device row"
        );
    }
    let ran = format!(
        "Backends run: {}",
        backends
            .iter()
            .map(|b| format!("{b} (f32, f64)"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(text.contains(&ran), "{ran}");
    let not_run: Vec<&str> = Backend::ALL
        .iter()
        .filter(|b| !backends.contains(b))
        .map(|b| b.name())
        .collect();
    println!(
        "backends run: {}; not run: {}",
        names.join(", "),
        not_run.join(", ")
    );
}
