//! `nd-fmm-bench`: the one-command FMM benchmark (Phase 4S T6). See the library's
//! documentation and `--help`; run it through `tools/bench/run.sh`.

use std::path::PathBuf;

use mpi::Threading;
use mpi::traits::Communicator;
use nd_fmm_bench::machine::{short_host, utc_now};
use nd_fmm_bench::measure::{SEED, run};
use nd_fmm_bench::options::{Command, HELP, USAGE, parse};
use nd_fmm_bench::problem::{
    check_output, compare_outputs, difference_line, errors_line, evaluate_problem, problem_line,
    write_full_output, write_problem,
};
use nd_fmm_bench::report::{Header, Report, summary_table};
use nd_fmm_exec::fmm::FmmBuilder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let options = match parse(&args[1..]) {
        Ok(Command::Run(options)) => *options,
        Ok(Command::Help) => {
            println!("{HELP}\n\n{USAGE}");
            return;
        }
        // The problem commands (Phase 6 T9) run without MPI.
        Ok(Command::WriteProblem(spec)) => {
            match write_problem(&spec) {
                Ok(written) => println!("{}", problem_line(&spec, &written)),
                Err(error) => {
                    eprintln!(
                        "nd-fmm-bench: cannot write {}: {error}",
                        spec.file.display()
                    );
                    std::process::exit(1);
                }
            }
            return;
        }
        Ok(Command::CheckOutput { problem, outputs }) => {
            let mut failed = false;
            for output in &outputs {
                match check_output(&problem, output) {
                    Ok(errors) => println!("{}", errors_line(output, &errors)),
                    Err(error) => {
                        eprintln!(
                            "nd-fmm-bench: cannot check {} against {}: {error}",
                            output.display(),
                            problem.display()
                        );
                        failed = true;
                    }
                }
            }
            std::process::exit(i32::from(failed));
        }
        Ok(Command::Compare { outputs }) => {
            let mut failed = false;
            for other in &outputs[1..] {
                match compare_outputs(&outputs[0], other) {
                    Ok(difference) => {
                        println!("{}", difference_line(&outputs[0], other, &difference));
                    }
                    Err(error) => {
                        eprintln!(
                            "nd-fmm-bench: cannot compare {} with {}: {error}",
                            other.display(),
                            outputs[0].display()
                        );
                        failed = true;
                    }
                }
            }
            std::process::exit(i32::from(failed));
        }
        // Our FMM on a problem file, on one rank (Phase 6 T9's cross-check).
        Ok(Command::WriteOutput {
            problem,
            output,
            degree,
            strategy,
            threads,
        }) => {
            let (universe, provided) =
                mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
            let comm = universe.world();
            let threads = threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            });
            if comm.size() != 1 || (threads > 1 && provided < Threading::Funneled) {
                eprintln!("nd-fmm-bench --write-output runs on one rank, with MPI at Funneled");
                std::process::exit(2);
            }
            let builder = FmmBuilder::<f64>::new(degree)
                .gradients(true)
                .threads(threads)
                .strategy(strategy);
            match evaluate_problem(&problem, &builder, &comm) {
                Ok((header, values, seconds)) => {
                    let (m, n) = (header.vectors, header.workload.n);
                    if let Err(error) = write_full_output(&output, m, n, &values) {
                        eprintln!("nd-fmm-bench: cannot write {}: {error}", output.display());
                        std::process::exit(1);
                    }
                    println!(
                        "output: file={} problem={} p={degree} strategy={strategy:?} \
                         threads={threads} vectors={m} points={n} evaluate_s={seconds:.3}",
                        output.display(),
                        problem.display()
                    );
                }
                Err(message) => {
                    eprintln!("nd-fmm-bench: {message}");
                    std::process::exit(1);
                }
            }
            return;
        }
        Err(message) => {
            eprintln!("nd-fmm-bench: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if comm.size() != 1 {
        eprintln!("nd-fmm-bench runs on one rank");
        std::process::exit(2);
    }
    if options.threads() > 1 && provided < Threading::Funneled {
        eprintln!(
            "{} threads need MPI at Funneled; it provides {provided:?}",
            options.threads()
        );
        std::process::exit(2);
    }
    let program = std::path::Path::new(&args[0]).file_name().map_or_else(
        || args[0].clone(),
        |name| name.to_string_lossy().into_owned(),
    );
    let command = std::iter::once(program)
        .chain(args[1..].iter().cloned())
        .collect::<Vec<_>>()
        .join(" ");
    let header = Header::collect(&options, command);
    let output = options.output.clone().unwrap_or_else(|| {
        let (_, stamp) = utc_now();
        PathBuf::from("bench-results").join(format!("{}-{stamp}.md", short_host(&header.host)))
    });
    let mut report = Report {
        header,
        options,
        seed: SEED,
        rows: Vec::new(),
    };
    eprintln!("nd-fmm-bench: writing {}", output.display());
    if let Err(error) = run(&mut report, &output, &comm) {
        eprintln!("nd-fmm-bench: cannot write {}: {error}", output.display());
        std::process::exit(1);
    }
    println!("{}", summary_table(&report.rows));
    println!("Report: {}", output.display());
}
