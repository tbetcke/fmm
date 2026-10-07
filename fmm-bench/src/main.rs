//! `nd-fmm-bench`: the one-command FMM benchmark (Phase 4S T6). See the library's
//! documentation and `--help`; run it through `tools/bench/run.sh`.

use std::path::PathBuf;

use mpi::Threading;
use mpi::traits::Communicator;
use nd_fmm_bench::machine::{short_host, utc_now};
use nd_fmm_bench::measure::{SEED, run};
use nd_fmm_bench::options::{Command, HELP, USAGE, parse};
use nd_fmm_bench::report::{Header, Report, summary_table};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let options = match parse(&args[1..]) {
        Ok(Command::Run(options)) => *options,
        Ok(Command::Help) => {
            println!("{HELP}\n\n{USAGE}");
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
