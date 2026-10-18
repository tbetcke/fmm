//! Prints the cost model of the stop rule (src/model.rs) as Markdown:
//! `cargo run --release -p nd-fmm-spike-plane-wave --example model`.

use nd_fmm_spike_plane_wave::geometry::{Counts, Level};
use nd_fmm_spike_plane_wave::model::{self, ROTATION_ONE_CORE_MEASURED};
use nd_fmm_spike_plane_wave::quadrature::Quadrature;

fn main() {
    println!("## Counts per level (uniform)\n");
    println!(
        "| level | boxes | V pairs | per box | shifts per box | accumulators per box | child adds per box | child shifts per box |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for l in 2..=6 {
        let c = Counts::of(&Level { level: l });
        let b = c.boxes as f64;
        println!(
            "| {l} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
            c.boxes,
            c.pairs,
            c.pairs as f64 / b,
            c.shifts as f64 / b,
            c.accumulators as f64 / b,
            c.child_adds as f64 / b,
            c.child_shifts as f64 / b
        );
    }
    println!(
        "\nV pairs of the cube's tree (levels 2–5): {}\n",
        model::cube_pairs()
    );

    println!("## Rotation per pair: derived against measured (µs, 1 × 1)\n");
    let all = model::measured();
    for (machine, p, t) in ROTATION_ONE_CORE_MEASURED {
        let derived = all
            .iter()
            .find(|m| m.machine == machine && m.p == p && m.f64 && m.split == "1 × 1")
            .unwrap()
            .rotation;
        println!(
            "- {} p = {p}: derived {derived:.3}, measured {t:.3}",
            machine.name()
        );
    }

    println!("\n## Quadratures\n");
    println!("| p | error | β = 0.1: s, H, FMM3D's Σ N, Σ F | β = 1: s, H |");
    println!("| --- | --- | --- | --- |");
    for p in [3, 6, 8, 12, 18] {
        let e = model::fmm_error(p);
        let (a, b) = (Quadrature::for_error(0.1 * e), Quadrature::for_error(e));
        println!(
            "| {p} | {e:.1e} | {} (measured {:.1e}), {}, {}, {} | {} (measured {:.1e}), {} |",
            a.nodes(),
            a.measured_error(),
            a.half_terms(),
            a.fmm3d_terms(),
            a.fourier_terms(),
            b.nodes(),
            b.measured_error(),
            b.half_terms()
        );
    }

    for beta in [0.1, 1.0, f64::NAN] {
        let rows = if beta.is_nan() {
            println!("\n## The M2L stage, cube N = 10⁶, the matched quadrature (ms)\n");
            model::rows_matched()
        } else {
            println!("\n## The M2L stage, cube N = 10⁶, β = {beta} (ms)\n");
            model::rows(beta)
        };
        println!(
            "| machine | precision, p | ranks × threads | bar | Batched fused | plane waves: optimistic / central / pessimistic | bar ÷ central | s, H | flops per box (level 5): rot / Fourier / physical / shifts |"
        );
        println!("| --- | --- | --- | --- | ---: | --- | ---: | --- | --- |");
        for r in rows {
            let m = &r.measured;
            println!(
                "| {} | {} p = {} | {} | {} {:.0} | {:.0} | {:.0} / {:.0} / {:.0} | **{:.2}** | {}, {} | {:.0}k / {:.0}k / {:.0}k / {:.0}k |",
                m.machine.name(),
                if m.f64 { "f64" } else { "f32" },
                m.p,
                m.split,
                r.bar.1,
                r.bar.0,
                r.fused,
                r.planewave[0],
                r.planewave[1],
                r.planewave[2],
                r.central_gain(),
                r.s,
                r.h,
                r.flops.rotations / 1e3,
                r.flops.fourier / 1e3,
                r.flops.physical / 1e3,
                r.flops.shifts / 1e3,
            );
        }
    }

    println!("\n## One H100, the matched quadrature (model)\n");
    println!(
        "| precision, p | dense M2L stage, measured (ms) | plane waves: GFLOP, traffic best / worst (GB) | compute-bound / memory-bound (ms) | dense ÷ the larger |"
    );
    println!("| --- | ---: | --- | --- | ---: |");
    for g in model::gpu_rows() {
        let t = g.planewave.0.max(g.planewave.1);
        println!(
            "| {} p = {} | {:.2} | {:.1}, {:.2} / {:.2} | {:.2} / {:.2} | {:.2} |",
            if g.f64 { "f64" } else { "f32" },
            g.p,
            g.dense,
            g.flops / 1e9,
            g.bytes.0 / 1e9,
            g.bytes.1 / 1e9,
            g.planewave.0,
            g.planewave.1,
            g.dense / t
        );
    }

    println!("\n## Traffic per box, level 5, the matched quadrature (KB)\n");
    println!("| p | plane waves best / worst | Batched's tables |");
    println!("| --- | --- | --- |");
    let c = Counts::of(&Level { level: 5 });
    for p in [3, 6, 8, 12, 18] {
        let q = model::matched(p);
        let (best, worst, tables) = model::traffic_per_box(&c, p, &q);
        println!(
            "| {p} | {:.1} / {:.1} | {:.1} |",
            best / 1e3,
            worst / 1e3,
            tables / 1e3
        );
    }
}
