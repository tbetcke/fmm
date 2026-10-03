//! Phase 4 T3 spike: device arithmetic per CubeCL backend. Prints Markdown on stdout and
//! progress on stderr. See docs/phase4/T3-device-arithmetic.md and REPORT.md.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-spike-device-arith --features cpu,metal -- \
//!     [--backends metal,cpu] [--sections primitives,compiler,domain,p2p,cpu-p2p,leafops] [--quick]
//! ```
//!
//! Metal needs a process with GPU access (outside the macOS sandbox). Timings are
//! reported, never asserted.

use std::time::Instant;

use nd_fmm_simd::P2pKernel;
use nd_fmm_spike_device_arith::backend::{Backend, describe, supports_f64};
use nd_fmm_spike_device_arith::real::Real;
use nd_fmm_spike_device_arith::{compiler, cpu_p2p, domain, leafops, p2p, pairs, primitives};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::p2p_kernels::{Form, Kernel, W1_SEED, W1_TARGETS, check, oracles, passes, w1};

/// The sizes of a run.
struct Sizes {
    f64_samples: usize,
    random_trials: usize,
    levels: Vec<u32>,
    term_pairs: usize,
    leaf_sizes: Vec<usize>,
    rate_leaves: usize,
    cpu_cells: Vec<usize>,
    cpu_leaves: usize,
    harmonic_points: usize,
}

impl Sizes {
    fn full() -> Self {
        Self {
            f64_samples: 10_000_000,
            random_trials: 2000,
            levels: pairs::all_levels(),
            term_pairs: 1_000_000,
            leaf_sizes: vec![8, 32, 64, 128],
            rate_leaves: 4096,
            cpu_cells: W1_TARGETS.to_vec(),
            cpu_leaves: 768,
            harmonic_points: 1000,
        }
    }

    fn quick() -> Self {
        Self {
            f64_samples: 100_000,
            random_trials: 100,
            levels: vec![0, 1, 8, 16],
            term_pairs: 20_000,
            leaf_sizes: vec![32],
            rate_leaves: 1024,
            cpu_cells: vec![24],
            cpu_leaves: 96,
            harmonic_points: 100,
        }
    }
}

struct Options {
    backends: Vec<Backend>,
    sections: Vec<String>,
    sizes: Sizes,
}

fn parse() -> Options {
    let mut o = Options {
        backends: Backend::runnable(),
        sections: [
            "primitives",
            "compiler",
            "domain",
            "p2p",
            "cpu-p2p",
            "leafops",
        ]
        .map(String::from)
        .to_vec(),
        sizes: Sizes::full(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--backends" => {
                o.backends = value()
                    .split(',')
                    .map(|b| Backend::parse(b).unwrap_or_else(|| panic!("unknown backend {b}")))
                    .collect()
            }
            "--sections" => o.sections = value().split(',').map(String::from).collect(),
            "--quick" => o.sizes = Sizes::quick(),
            other => panic!("unknown option {other}"),
        }
    }
    o
}

fn u(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.3}")
    } else {
        "∞".into()
    }
}

fn p2(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        format!("{x}")
    } else {
        format!("2^{:.2}", x.log2())
    }
}

fn section(title: &str) {
    println!("\n## {title}\n");
    eprintln!("== {title}");
}

fn primitives_for<T: Real>(client: &cubecl::prelude::Client, backend: Backend, sizes: &Sizes) {
    let mut sets = Vec::new();
    if T::BITS == 24 {
        let e = primitives::exhaustive_f32();
        sets.push(primitives::Inputs {
            label: e.label,
            a: e.a.iter().map(|&v| T::narrow(f64::from(v))).collect(),
            x: e.x.iter().map(|&v| T::narrow(f64::from(v))).collect(),
        });
    } else {
        sets.push(primitives::log_uniform::<T>(sizes.f64_samples, 7));
    }
    sets.push(primitives::powers_of_two::<T>(5));
    println!("| backend | precision | operation | inputs | max error (u_T) | at x | ≠ host |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for inputs in &sets {
        for op in primitives::Op::ALL {
            let a = primitives::measure(client, backend, op, inputs);
            println!(
                "| {} | {} | `{}` | {} | {} | {:.6e} | {} of {} |",
                backend.name(),
                T::NAME,
                op.name(),
                inputs.label,
                u(a.max_error),
                a.worst_x,
                a.differ,
                a.count
            );
        }
    }
    println!(
        "\n\"≠ host\": results whose bits differ from the correctly rounded host value \
         (`sqrt`, divisions) or from the host's fl(1/fl(√x)) (inverse square roots).\n"
    );
    println!("Edge cases ({}, {}; a = 1):\n", backend.name(), T::NAME);
    let names: Vec<String> = primitives::Op::ALL
        .iter()
        .map(|o| format!("`{}`", o.name()))
        .collect();
    println!("| x | {} |", names.join(" | "));
    println!("|{}", " --- |".repeat(names.len() + 1));
    for (x, cells) in primitives::edge_cases::<T>(client, backend) {
        println!("| {x} | {} |", cells.join(" | "));
    }
}

fn compiler_for<T: Real>(client: &cubecl::prelude::Client, backend: Backend) {
    println!("| backend | precision | probe | device | IEEE as written | verdict |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for (p, y, verdict) in compiler::run::<T>(client) {
        println!(
            "| {} | {} | {} | {:e} | {:e} | {} |",
            backend.name(),
            T::NAME,
            p.name,
            y.widen(),
            p.ieee.widen(),
            verdict
        );
    }
}

fn domain_for<T: Real>(client: &cubecl::prelude::Client, backend: Backend, sizes: &Sizes) -> bool {
    let mut ok = true;
    for (label, list) in [
        (
            "adversarial",
            pairs::adversarial::<T>(20261006, &sizes.levels),
        ),
        ("random", pairs::random::<T>(20261007, sizes.random_trials)),
    ] {
        let t = Instant::now();
        let dev = domain::run(client, backend, &list);
        let o = domain::check(&list, &dev);
        eprintln!(
            "  {label} {}: {} pairs in {:.1} s",
            T::NAME,
            list.len(),
            t.elapsed().as_secs_f64()
        );
        println!(
            "{} {} {label} pairs: {} pairs, {} coincident ({} from distinct points); ŷ ≠ host in {} \
             components, d ≠ host in {} (unfused ŷ: {}), d = 0 wrong in {}; smallest nonzero |d| {}. \
             On the host, reassociating d to (u_t − ĉ) − r̂ u_s would change {} components and gain or \
             lose a zero in {}.\n",
            backend.name(),
            T::NAME,
            o.pairs,
            o.coincident,
            o.coincident_distinct,
            o.y_differs,
            o.d_differs,
            o.d_unfused_differs,
            o.d_zero_wrong,
            p2(o.min_d),
            o.reassociation_sensitive,
            o.reassociation_breaks_zero
        );
        println!(
            "| r² formulation | rule holds | r² = 0 wrong | below 2^-106 | above 2^7 | smallest nonzero r² | largest r² | = host plain | = host fma forward |"
        );
        println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
        for (f, name) in domain::FORMULATIONS.iter().enumerate() {
            ok &= o.holds(f);
            println!(
                "| {name} | {} | {} | {} | {} | {} | {:.4} | {} | {} |",
                if o.holds(f) { "yes" } else { "**NO**" },
                o.r2_zero_wrong[f],
                o.r2_below[f],
                o.r2_above[f],
                p2(o.min_r2[f]),
                o.max_r2[f],
                o.r2_equals_plain[f],
                o.r2_equals_forward[f]
            );
        }
        println!();
    }
    ok
}

fn p2p_for<T: Real>(client: &cubecl::prelude::Client, backend: Backend, sizes: &Sizes) -> bool {
    let mut ok = true;
    let pairs = p2p::domain_pairs::<T>(sizes.term_pairs, 1, 3);
    let reference = p2p::reference_terms(&pairs);
    println!(
        "Pair terms against `nd_fmm_ref::p2p` ({} pairs over the kernel domain; gradient over r² ≥ {}), \
         and sums against `direct_sum` on W1 (worst over n_t ∈ {:?}, {} sets each, with gradients; \
         relative to the term magnitudes; the reference's own error in brackets):\n",
        pairs.len(),
        if T::BITS == 24 { "2^-84" } else { "2^-108" },
        sizes.leaf_sizes,
        nd_fmm_validate::p2p_kernels::CHECKED_SETS
    );
    println!(
        "| backend | precision | candidate | φ term (u_T) | ∇φ term (u_T) | φ, ∇φ term, nonzero |dₖ| ≥ 2^-53 (u_T) | φ sum | ∇φ sum | sum check |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for c in p2p::Candidate::all() {
        let e = p2p::pair_errors(client, backend, c, &pairs, &reference);
        let (mut acc, mut refacc) = (Default::default(), Default::default());
        let mut pass = true;
        for &n_t in &sizes.leaf_sizes {
            let (a, r) = p2p::leaf_check::<T>(client, backend, c, n_t);
            pass &= passes::<T>(a, r);
            acc = nd_fmm_validate::p2p_kernels::Accuracy::worst(acc, a);
            refacc = nd_fmm_validate::p2p_kernels::Accuracy::worst(refacc, r);
        }
        ok &= pass;
        println!(
            "| {} | {} | {} | {} | {} | {}, {} | {:.2e} [{:.2e}] | {:.2e} [{:.2e}] | {} |",
            backend.name(),
            T::NAME,
            c.name(),
            u(e.potential),
            u(e.gradient),
            u(e.potential_grid),
            u(e.gradient_grid),
            acc.max_potential,
            refacc.max_potential,
            acc.max_gradient,
            refacc.max_gradient,
            if pass { "pass" } else { "**FAIL**" }
        );
    }
    if backend.is_gpu() {
        println!(
            "\nQuick throughput, Gpairs/s ({} target leaves per launch, one unit per target, 64 per cube, \
             launches queued between syncs, compilation excluded, median of 15 batches of at least 20 ms):\n",
            sizes.rate_leaves
        );
        let cols: Vec<String> = sizes
            .leaf_sizes
            .iter()
            .map(|n| format!("n_t = {n}"))
            .collect();
        println!(
            "| backend | precision | candidate | output | {} |",
            cols.join(" | ")
        );
        println!("|{}", " --- |".repeat(4 + cols.len()));
        for c in p2p::Candidate::all() {
            for gradients in [false, true] {
                let rates: Vec<String> = sizes
                    .leaf_sizes
                    .iter()
                    .map(|&n_t| {
                        format!(
                            "{:.1}",
                            p2p::leaf_rate::<T>(client, c, n_t, gradients, sizes.rate_leaves) / 1e9
                        )
                    })
                    .collect();
                println!(
                    "| {} | {} | {} | {} | {} |",
                    backend.name(),
                    T::NAME,
                    c.name(),
                    if gradients { "φ, ∇φ" } else { "φ" },
                    rates.join(" | ")
                );
            }
        }
    }
    ok
}

/// One cell of the CPU-runtime comparison: (seconds per launch at 1 and 12 units, and of
/// `nd-fmm-simd` at 1 and 12 threads), or `None` if the direct-sum check failed.
fn cpu_cell<T: Real>(
    client: &cubecl::prelude::Client,
    n_t: usize,
    gradients: bool,
    leaves: usize,
    identical: &mut (usize, usize),
) -> Option<[f64; 4]> {
    let pool = w1::<T>(n_t, W1_SEED);
    let oracles = oracles(&pool);
    let checked = &pool[..oracles.len()];
    let reference = check(
        &Kernel::<T>::Reference,
        checked,
        &oracles,
        Form::Gathered,
        gradients,
    );
    let acc = cpu_p2p::accuracy_on(client, checked, &oracles, gradients, 1);
    let acc12 = cpu_p2p::accuracy_on(client, checked, &oracles, gradients, 12);
    if !passes::<T>(acc, reference) || !passes::<T>(acc12, reference) {
        println!(
            "| {} | {} | {n_t} | check FAILED: {:.2e} (reference {:.2e}); row not timed | | | | | | |",
            T::NAME,
            if gradients { "φ, ∇φ" } else { "φ" },
            acc.max().max(acc12.max()),
            reference.max()
        );
        return None;
    }
    let (same, total) =
        cpu_p2p::identical_to_simd(client, P2pKernel::<T>::detect(), checked, gradients);
    eprintln!("    bit-identical to nd-fmm-simd: {same} of {total}");
    identical.0 += same;
    identical.1 += total;
    let device = cpu_p2p::upload(client, &pool);
    let kernel = P2pKernel::<T>::detect();
    Some([
        cpu_p2p::time_launch::<T>(client, &device, leaves, gradients, 1),
        cpu_p2p::time_launch::<T>(client, &device, leaves, gradients, 12),
        cpu_p2p::time_simd(kernel, &pool, leaves, gradients, 1),
        cpu_p2p::time_simd(kernel, &pool, leaves, gradients, 12),
    ])
}

fn cpu_p2p_for<T: Real>(
    client: &cubecl::prelude::Client,
    sizes: &Sizes,
    ratios: &mut Vec<(f64, f64)>,
) -> bool {
    let mut ok = true;
    let (n, k) = cpu_p2p::layout::<T>();
    let pairs = p2p::domain_pairs::<T>(sizes.term_pairs, n * k, 9);
    let e = cpu_p2p::pair_errors(client, &pairs);
    println!(
        "{}: N = {n}, K = {k}; pair terms over the kernel domain ({} pairs): φ {} u_T, ∇φ {} u_T.\n",
        T::NAME,
        e.pairs,
        u(e.potential),
        u(e.gradient)
    );
    let mut identical = (0, 0);
    println!(
        "| precision | output | n_t | CPU runtime 1 unit (Gpairs/s) | nd-fmm-simd 1 thread | ratio (time) | CPU runtime 12 units | nd-fmm-simd 12 threads | ratio (time) |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for gradients in [false, true] {
        for &n_t in &sizes.cpu_cells {
            eprintln!("  {} n_t = {n_t} gradients = {gradients}", T::NAME);
            let Some([c1, c12, s1, s12]) =
                cpu_cell::<T>(client, n_t, gradients, sizes.cpu_leaves, &mut identical)
            else {
                ok = false;
                continue;
            };
            let pairs = (sizes.cpu_leaves * n_t * 27 * n_t) as f64;
            ratios.push((c1 / s1, c12 / s12));
            println!(
                "| {} | {} | {n_t} | {:.3} | {:.3} | {:.2} | {:.3} | {:.3} | {:.2} |",
                T::NAME,
                if gradients { "φ, ∇φ" } else { "φ" },
                pairs / c1 / 1e9,
                pairs / s1 / 1e9,
                c1 / s1,
                pairs / c12 / 1e9,
                pairs / s12 / 1e9,
                c12 / s12
            );
        }
    }
    println!(
        "\n{}: outputs bit-identical to `nd-fmm-simd` ({}) on the checked sets of every cell: {} of {}.\n",
        T::NAME,
        P2pKernel::<T>::detect().isa(),
        identical.0,
        identical.1
    );
    ok
}

fn overheads(client: &cubecl::prelude::Client) {
    let pool = w1::<f32>(64, W1_SEED);
    let device = cpu_p2p::upload(client, &pool);
    println!("| launch | units | queued (µs per launch) | launch + sync (µs) |");
    println!("| --- | --- | --- | --- |");
    for (label, leaves) in [
        ("empty (no leaves)", 0),
        ("one target leaf, n_t = 64, f32 φ", 1),
    ] {
        for units in [1, 12] {
            let queued = cpu_p2p::time_launch::<f32>(client, &device, leaves, false, units);
            let latency = cpu_p2p::launch_latency::<f32>(client, &device, leaves, units);
            println!(
                "| {label} | {units} | {:.1} | {:.1} |",
                queued * 1e6,
                latency * 1e6
            );
        }
    }
    println!();
}

fn leafops_for<T: Real>(client: &cubecl::prelude::Client, backend: Backend, sizes: &Sizes) {
    for p in [8, 20] {
        for irregular in [false, true] {
            let r = leafops::harmonics::<T>(client, backend, p, irregular, sizes.harmonic_points);
            println!(
                "| {} | {} | {} harmonics, p = {p} | {} of {} | {} ({} where normal; {} host values subnormal) | {} | {} |",
                backend.name(),
                T::NAME,
                if irregular { "irregular" } else { "regular" },
                r.identical,
                r.values,
                u(r.vs_host),
                u(r.vs_host_normal),
                r.subnormal,
                u(r.device_vs_f64),
                u(r.host_vs_f64)
            );
        }
    }
    let g = leafops::gemm::<T>(client, backend, 8, 512);
    println!(
        "| {} | {} | y += A x, n = 81, 512 columns | {} of {} (= fused host: {}) | {} | – | – |",
        backend.name(),
        T::NAME,
        g.equals_apply,
        g.outputs,
        g.equals_fused,
        u(g.vs_apply)
    );
}

fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

fn main() {
    let o = parse();
    let has = |s: &str| o.sections.iter().any(|x| x == s);
    println!("# Device arithmetic (Phase 4 T3): raw output\n");
    println!(
        "- Machine: {} ({} cores; performance cores {:?})",
        cpu_model(),
        cores(),
        performance_cores()
    );
    println!("- Build: {}, {}", target(), toolchain());
    println!("- CubeCL 0.11.0-pre.4 (pinned)");
    let blas: Vec<String> = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ]
    .iter()
    .map(|v| {
        format!(
            "{v}={}",
            std::env::var(v).unwrap_or_else(|_| "unset".into())
        )
    })
    .collect();
    println!("- Environment: {}", blas.join(", "));
    println!("- Sections: {}", o.sections.join(", "));
    let mut clients = Vec::new();
    for &b in &o.backends {
        let client = b.client();
        println!("- {}", describe(&client, b));
        clients.push((b, client));
    }
    let mut failures = Vec::new();
    for (b, client) in &clients {
        let (b, client) = (*b, client);
        let f64_ok = supports_f64(client);
        if has("primitives") {
            section(&format!("Primitive accuracy: {}", b.name()));
            primitives_for::<f32>(client, b, &o.sizes);
            if f64_ok {
                primitives_for::<f64>(client, b, &o.sizes);
            }
        }
        if has("compiler") {
            section(&format!("Compiler behaviour: {}", b.name()));
            compiler_for::<f32>(client, b);
            if f64_ok {
                compiler_for::<f64>(client, b);
            }
        }
        if has("domain") {
            section(&format!("The §3.13 argument on the device: {}", b.name()));
            if !domain_for::<f32>(client, b, &o.sizes) {
                failures.push(format!("{} f32: §3.13 rule", b.name()));
            }
            if f64_ok && !domain_for::<f64>(client, b, &o.sizes) {
                failures.push(format!("{} f64: §3.13 rule", b.name()));
            }
        }
        if has("p2p") {
            section(&format!("P2P candidates: {}", b.name()));
            if !p2p_for::<f32>(client, b, &o.sizes) {
                failures.push(format!("{} f32: P2P sum check", b.name()));
            }
            if f64_ok && !p2p_for::<f64>(client, b, &o.sizes) {
                failures.push(format!("{} f64: P2P sum check", b.name()));
            }
        }
        if has("cpu-p2p") && b == Backend::Cpu {
            section("CPU-shaped P2P on the CPU runtime against nd-fmm-simd");
            println!(
                "W1 gathered, {} target leaves per launch (the 64-set pool cycled); 1 unit against `P2pKernel::detect()` ({}) \
                 on one thread, 12 units against it on 12 rayon threads; kernel compilation excluded; median of 15 \
                 batches of at least 20 ms. Ratio = CPU-runtime time / nd-fmm-simd time.\n",
                o.sizes.cpu_leaves,
                P2pKernel::<f32>::detect().isa()
            );
            let mut ratios = Vec::new();
            let ok32 = cpu_p2p_for::<f32>(client, &o.sizes, &mut ratios);
            let ok64 = cpu_p2p_for::<f64>(client, &o.sizes, &mut ratios);
            if !(ok32 && ok64) {
                failures.push("cpu f32/f64: CPU-shaped P2P direct-sum check".into());
            }
            if !ratios.is_empty() {
                let one: Vec<f64> = ratios.iter().map(|r| r.0).collect();
                let all: Vec<f64> = ratios.iter().map(|r| r.1).collect();
                println!(
                    "Geometric mean over {} cells: one thread {:.3}, all performance cores {:.3}. Rule: at most 1.5 sets a \
                     CPU-runtime target; above 1.5 the CPU runtime stays correctness-only.\n",
                    ratios.len(),
                    geomean(&one),
                    geomean(&all)
                );
            }
            println!("Per-launch overhead (f32, φ):\n");
            overheads(client);
        }
        if has("leafops") {
            section(&format!("Harmonics and GEMM: {}", b.name()));
            println!(
                "| backend | precision | test | bit-identical to host | max diff from host (u_T) | device vs f64 (u_T) | host vs f64 (u_T) |"
            );
            println!("| --- | --- | --- | --- | --- | --- | --- |");
            leafops_for::<f32>(client, b, &o.sizes);
            if f64_ok {
                leafops_for::<f64>(client, b, &o.sizes);
            }
            println!(
                "\nHarmonics errors per point and degree, relative to the largest |value| of that degree; GEMM relative to max |y|."
            );
        }
    }
    println!("\n## Summary\n");
    if failures.is_empty() {
        println!(
            "Every check passed on: {}.",
            o.backends
                .iter()
                .map(|b| b.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        println!("Failed: {}.", failures.join("; "));
        std::process::exit(1);
    }
}
