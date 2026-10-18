//! Times the M2L of one uniform level: `Batched` (separate and fused), `Rotation` and
//! plane waves (merge-and-shift, and direct), f64, one thread. The multipoles are those
//! of 4 random charges per box (uniform in the box, charges in [0, 1)). With `--check` it
//! also compares the plane-wave locals with `Batched`'s as potentials: the largest
//! |φ_pw − φ_batched| at 2 random points per box over the largest |φ_batched|.
//!
//! ```text
//! cargo run --release -p nd-fmm-spike-plane-wave --example level -- \
//!     --level 4 --p 8 --s 17 [--reps 5] [--variants batched,fused,rotation,merge,direct] [--check]
//!     [--isa scalar|neon|avx2|avx512]
//! ```
//!
//! `--isa` names the column kernel's ISA of `Batched` (default `Isa::detect`: AVX2 on an
//! AVX-512 machine).
//!
//! One line per variant: `variant p s level pairs median_ms us_per_pair [parts]`.
//! tools/run_node.sh runs one copy per core at once (the node's processes, as T1 did).

use std::time::Instant;

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_simd::{Isa, MulAdd};
use nd_fmm_spike_plane_wave::baseline::{BatchedM2l, Batches, RotationM2l};
use nd_fmm_spike_plane_wave::geometry::Level;
use nd_fmm_spike_plane_wave::planewave::{
    LevelPlan, Parts, PlaneWave, Scratch, Translation, level_m2l,
};
use nd_fmm_spike_plane_wave::quadrature::Quadrature;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let level = Level {
        level: value("--level").map_or(4, |v| v.parse().unwrap()),
    };
    let p: usize = value("--p").expect("--p").parse().unwrap();
    let s: usize = value("--s").expect("--s").parse().unwrap();
    let reps: usize = value("--reps").map_or(5, |v| v.parse().unwrap());
    let variants = value("--variants").unwrap_or_else(|| "batched,fused,rotation,merge".into());
    let check = args.iter().any(|a| a == "--check");
    let isa = value("--isa").map(|v| match v.as_str() {
        "scalar" => Isa::Scalar,
        "neon" => Isa::Neon,
        "avx2" => Isa::Avx2,
        "avx512" => Isa::Avx512,
        other => panic!("unknown ISA {other}"),
    });

    let n = (p + 1) * (p + 1);
    let pairs = level.pairs();
    let npairs = pairs.len();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    // M̃ = Σ q conj Rₙᵐ(u), u = (y − c)/r with r = ½ box width (CONVENTIONS §3.7).
    let layout = Layout::new(p);
    let mut multipoles = vec![0.0; level.boxes() * n];
    let mut r = vec![0.0; n];
    for b in 0..level.boxes() {
        for _ in 0..4 {
            let u: [f64; 3] = core::array::from_fn(|_| 2.0 * uniform() - 1.0);
            let q = uniform();
            harmonics::regular(p, u, &mut r);
            for (i, (&ri, m)) in r
                .iter()
                .zip(&mut multipoles[b * n..(b + 1) * n])
                .enumerate()
            {
                let order = i as isize - layout.idx(i.isqrt(), 0) as isize;
                *m += q * if order < 0 { -ri } else { ri };
            }
        }
    }
    let points: Vec<[f64; 3]> = (0..2 * level.boxes())
        .map(|_| core::array::from_fn(|_| 2.0 * uniform() - 1.0))
        .collect();
    // r φ of a box's local at v (scaled coordinates).
    let potential = |local: &[f64], v: [f64; 3], r: &mut [f64]| {
        harmonics::regular(p, v, r);
        let mut sum = 0.0;
        for nn in 0..=p {
            sum += local[layout.idx(nn, 0)] * r[layout.idx(nn, 0)];
            for m in 1..=nn as isize {
                let (a, b) = (layout.idx(nn, m), layout.idx(nn, -m));
                sum += 2.0 * (local[a] * r[a] - local[b] * r[b]);
            }
        }
        sum
    };
    let mut reference: Option<Vec<f64>> = None;
    let report = |name: &str, times: Vec<f64>, extra: String| {
        let ms = median(times) * 1e3;
        println!(
            "{name} p={p} s={s} level={} pairs={npairs} median_ms={ms:.3} us_per_pair={:.4}{extra}",
            level.level,
            ms * 1e3 / npairs as f64
        );
    };

    for variant in variants.split(',') {
        let mut locals = vec![0.0; level.boxes() * n];
        match variant {
            "batched" | "fused" => {
                let mode = if variant == "fused" {
                    MulAdd::Fused
                } else {
                    MulAdd::Separate
                };
                let mut op = BatchedM2l::new(p, mode, isa);
                let batches = Batches::new(&pairs);
                let mut times = Vec::new();
                for _ in 0..=reps {
                    locals.fill(0.0);
                    let t = Instant::now();
                    op.apply(&level, &batches, &multipoles, &mut locals);
                    times.push(t.elapsed().as_secs_f64());
                }
                times.remove(0);
                report(variant, times, format!(" isa={:?}", op.kernel().isa()));
                if variant == "batched" {
                    reference = Some(locals.clone());
                }
            }
            "rotation" => {
                let mut op = RotationM2l::new(p);
                let mut times = Vec::new();
                for _ in 0..=reps {
                    locals.fill(0.0);
                    let t = Instant::now();
                    op.apply(&pairs, &multipoles, &mut locals);
                    times.push(t.elapsed().as_secs_f64());
                }
                times.remove(0);
                report(variant, times, String::new());
            }
            "merge" | "direct" => {
                let translation = if variant == "merge" {
                    Translation::MergeShift
                } else {
                    Translation::Direct
                };
                let op = PlaneWave::new(p, Quadrature::with_nodes(s));
                let plan = LevelPlan::new(&op, level, translation);
                let mut scratch = Scratch::new(&op, level);
                let (mut times, mut parts) = (Vec::new(), Vec::<Parts>::new());
                for _ in 0..=reps {
                    locals.fill(0.0);
                    let t = Instant::now();
                    let part = level_m2l(&op, &plan, &mut scratch, &multipoles, &mut locals);
                    times.push(t.elapsed().as_secs_f64());
                    parts.push(part);
                }
                times.remove(0);
                parts.remove(0);
                let out = median(parts.iter().map(|x| x.outgoing).collect()) * 1e3;
                let inc = median(parts.iter().map(|x| x.incoming).collect()) * 1e3;
                let mut extra = format!(
                    " h={} waves_mb={:.1} tables={} outgoing_ms={out:.3} incoming_ms={inc:.3}",
                    op.h(),
                    scratch.waves.len() as f64 * 8.0 / 1e6,
                    plan.table_count()
                );
                if check && let Some(reference) = &reference {
                    let (mut diff, mut largest): (f64, f64) = (0.0, 0.0);
                    for (i, &v) in points.iter().enumerate() {
                        let b = i / 2;
                        let pw = potential(&locals[b * n..(b + 1) * n], v, &mut r);
                        let ba = potential(&reference[b * n..(b + 1) * n], v, &mut r);
                        diff = diff.max((pw - ba).abs());
                        largest = largest.max(ba.abs());
                    }
                    extra += &format!(" potential_vs_batched={:.2e}", diff / largest);
                }
                report(variant, times, extra);
            }
            other => panic!("unknown variant {other}"),
        }
    }
}
