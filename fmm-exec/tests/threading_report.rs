//! `ThreadingReport` against the BLAS thread variables (C3.5): with every variable unset,
//! set to 1, set to 4, and set to a mix, the report shows each value, and with more than
//! one rayon thread lists as a warning exactly the variables that are unset or not 1.
//!
//! A test must not set environment variables in its own process (the crate never sets
//! one, and `std::env::set_var` is unsafe in Rust 2024). The test therefore starts this
//! executable again for each case, as a child process with its own environment and the
//! marker variable [`CHILD`]; the child reads the report and checks it, and the parent
//! checks that every child passed. No MPI: the report only reads the environment.
//! Error measure: exact equality of the values and of the warning lists.

use std::process::Command;

use mpi::Threading;
use nd_fmm_exec::threading::{BLAS_VARIABLES, ThreadingReport};

/// The marker variable of a child process; its value names the case.
const CHILD: &str = "ND_FMM_EXEC_THREADING_REPORT_CASE";

/// The name of the test, which the child runs alone.
const TEST: &str = "report_shows_each_variable_and_the_warning_rule";

/// The cases: a name and the value of each of `BLAS_VARIABLES`, `None` for unset.
const CASES: [(&str, [Option<&str>; 5]); 4] = [
    ("unset", [None; 5]),
    ("one", [Some("1"); 5]),
    ("four", [Some("4"); 5]),
    (
        "mixed",
        [Some("1"), Some("4"), None, Some(" 1 "), Some("0")],
    ),
];

#[test]
fn report_shows_each_variable_and_the_warning_rule() {
    if let Ok(case) = std::env::var(CHILD) {
        check_child(&case);
        return;
    }
    let exe = std::env::current_exe().expect("the test executable");
    for (case, values) in CASES {
        let mut command = Command::new(&exe);
        command
            .args([TEST, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, case);
        for (name, value) in BLAS_VARIABLES.iter().zip(values) {
            match value {
                Some(value) => command.env(name, value),
                None => command.env_remove(name),
            };
        }
        let output = command.output().expect("the child process starts");
        let (stdout, stderr) = (
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            output.status.success(),
            "case {case}: the child failed\n{stdout}\n{stderr}"
        );
        assert!(
            stdout.contains(&format!("case {case} checked")),
            "case {case}: the child did not run the check\n{stdout}\n{stderr}"
        );
        for line in stdout.lines().filter(|l| l.starts_with("report ")) {
            eprintln!("{case}: {line}");
        }
    }
}

/// The child's side: reads the report at one and at four threads and checks it against
/// the case's values.
fn check_child(case: &str) {
    let (_, values) = CASES
        .iter()
        .find(|(name, _)| *name == case)
        .unwrap_or_else(|| panic!("unknown case {case}"));
    for threads in [1, 4] {
        let report = ThreadingReport::read(threads, Threading::Funneled);
        assert_eq!(report.threads, threads);
        assert_eq!(report.mpi, Threading::Funneled);
        let mut expected = Vec::new();
        for ((variable, name), value) in report.blas.iter().zip(BLAS_VARIABLES).zip(values) {
            assert_eq!(variable.name, name);
            assert_eq!(variable.value.as_deref(), *value, "{name}");
            let one = value.is_some_and(|v| v.trim() == "1");
            assert_eq!(variable.is_one(), one, "{name}");
            if threads > 1 && !one {
                expected.push(name);
            }
        }
        let warnings: Vec<&str> = report.warnings().iter().map(|v| v.name).collect();
        assert_eq!(warnings, expected, "{threads} threads");
        let line = report.to_string();
        assert_eq!(line.matches("(warning)").count(), expected.len(), "{line}");
        for (name, value) in BLAS_VARIABLES.iter().zip(values) {
            let shown = match value {
                Some(value) => format!("{name}={value}"),
                None => format!("{name} unset"),
            };
            assert!(line.contains(&shown), "{line} lacks {shown}");
        }
        println!("report {line}");
    }
    println!("case {case} checked");
}
