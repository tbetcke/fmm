//! Runs every variant on every workload cell: the `direct_sum` check first, then the
//! timing (design simd-p2p.md §8.1). A prototype that fails its check is not timed.

use std::time::Instant;

use crate::green::{Gk, unpack};
use crate::kernels::{Order, Proto, Prototypes};
use crate::machine::median_time_per_call;
use crate::simd::Elem;
use crate::workloads::{Accuracy, Cell, Form, Oracle, Set, accuracy, w1, w2};

/// What is run.
pub enum Variant<E> {
    /// `nd_fmm_ref::p2p::p2p`, the Phase 3 path.
    Reference,
    /// green-kernels' `evaluate_st`.
    Green,
    /// A spike prototype.
    Proto(Proto<E>),
}

/// One measured row.
#[derive(Clone, Debug)]
pub struct Row {
    /// Workload cell.
    pub cell: Cell,
    /// "f32" or "f64".
    pub prec: &'static str,
    /// With gradients.
    pub grad: bool,
    /// "reference", "green-kernels" or the prototype's entry point.
    pub label: String,
    /// "scalar", "neon", "avx2", or green-kernels' pulp backend.
    pub isa: String,
    /// Loop order of a prototype.
    pub order: Option<Order>,
    /// K or T of a prototype.
    pub block: usize,
    /// "best", "gk" or "relaxed" for a prototype.
    pub flavour: &'static str,
    /// Inverse square root of a prototype.
    pub rsqrt: &'static str,
    /// Model operations per pair and lane (design §4.2) of a prototype.
    pub model_ops: usize,
    /// Lanes of a prototype.
    pub lanes: usize,
    /// Pairs per second; `None` if the check failed.
    pub pairs_per_s: Option<f64>,
    /// Worst accuracy over the checked sets.
    pub acc: Accuracy,
    /// Whether the check passed.
    pub passed: bool,
    /// Per-pair rows: whether the result equals the gathered call bit for bit.
    pub same_as_gathered: Option<bool>,
}

/// Number of sets of a pool that the check runs on.
const CHECKED_SETS: usize = 4;

/// The outputs of a set: potentials, gradients and green-kernels' interleaved values.
type Outputs<'a, E> = (&'a mut [E], &'a mut [[E; 3]], &'a mut [E]);

fn one<E: Elem + Gk>(
    v: &Variant<E>,
    grad: bool,
    s: &[[E; 3]],
    q: &[E],
    t: &[[E; 3]],
    (pot, g, gk): Outputs<'_, E>,
) {
    match v {
        Variant::Reference => nd_fmm_ref::p2p::p2p(s, q, t, pot, grad.then_some(g)),
        Variant::Green => {
            let n = t.len() * if grad { 4 } else { 1 };
            E::gk(grad, s, q, t, &mut gk[..n]);
        }
        Variant::Proto(p) => p.call(s, q, t, pot, grad.then_some(g)),
    }
}

/// One evaluation of a set in the given form, adding into its outputs.
pub fn call<E: Elem + Gk>(v: &Variant<E>, set: &mut Set<E>, form: Form, grad: bool) {
    let Set {
        targets,
        leaves,
        sources,
        charges,
        pot,
        grad: g,
        gk,
    } = set;
    match form {
        Form::Gathered => one(v, grad, sources, charges, targets, (pot, g, gk)),
        Form::PerPair => {
            for (s, q) in leaves.iter() {
                one(v, grad, s, q, targets, (&mut *pot, &mut *g, &mut *gk));
            }
        }
    }
}

/// The accuracy of a variant's outputs on a set.
fn measure<E: Elem>(v: &Variant<E>, set: &Set<E>, o: &Oracle, grad: bool) -> Accuracy {
    if let Variant::Green = v {
        let (pot, g) = unpack(grad, &set.gk, set.targets.len());
        accuracy::<f64>(o, &pot, g.as_deref())
    } else {
        accuracy(o, &set.pot, grad.then_some(&set.grad[..]))
    }
}

fn same_bits<E: Elem>(a: &[E], b: &[E]) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (*x).to_f64().to_bits() == (*y).to_f64().to_bits())
}

/// Checks and times one variant on a pool.
fn run_variant<E: Elem + Gk>(
    v: &Variant<E>,
    cell: Cell,
    pool: &mut [Set<E>],
    oracles: &[Oracle],
    grad: bool,
    gk_backend: &str,
    reference: Option<Accuracy>,
) -> Row {
    let form = cell.form();
    let mut acc = Accuracy::default();
    for (set, o) in pool.iter_mut().zip(oracles) {
        set.clear();
        call(v, set, form, grad);
        acc = acc.worst(measure(v, set, o, grad));
    }
    let n_s = pool[0].sources.len();
    let relaxed = matches!(v, Variant::Proto(p) if p.flavour == "relaxed");
    // Requirement 2 states the sum tolerance up to 4,096 sources. Beyond (W2 at
    // N = 10⁴) a row also passes within twice the reference's own error on the cell.
    let passed = acc.passes::<E>(n_s, relaxed)
        || (n_s > 4096
            && !relaxed
            && !acc.non_finite
            && reference.is_some_and(|r| acc.max() <= 2.0 * r.max()));
    let same_as_gathered = match (v, form) {
        (Variant::Proto(_), Form::PerPair) => {
            let set = &mut pool[0];
            set.clear();
            call(v, set, Form::PerPair, grad);
            let (p, g) = (set.pot.clone(), set.grad.clone());
            set.clear();
            call(v, set, Form::Gathered, grad);
            Some(same_bits(&p, &set.pot) && same_bits(g.as_flattened(), set.grad.as_flattened()))
        }
        _ => None,
    };
    // A failed check aborts a prototype's row. The reference and green-kernels are
    // baselines and are timed regardless; their accuracy is reported.
    let time = !matches!(v, Variant::Proto(_)) || passed;
    let pairs_per_s = time.then(|| {
        let pairs = pool[0].pairs() as f64;
        let len = pool.len();
        let t = median_time_per_call(|calls| {
            let start = Instant::now();
            for c in 0..calls {
                call(v, &mut pool[c % len], form, grad);
            }
            start.elapsed()
        });
        pairs / t
    });
    let (label, isa, order, block, flavour, rsqrt, model_ops, lanes) = match v {
        Variant::Reference => (
            "reference".to_string(),
            "scalar".to_string(),
            None,
            0,
            "",
            "sqrt+div",
            0,
            1,
        ),
        Variant::Green => (
            "green-kernels".to_string(),
            gk_backend.to_string(),
            None,
            0,
            "",
            "",
            0,
            0,
        ),
        Variant::Proto(p) => (
            p.name.to_string(),
            p.isa.to_string(),
            Some(p.order),
            p.block,
            p.flavour,
            p.rsqrt,
            p.model_ops(),
            p.lanes,
        ),
    };
    Row {
        cell,
        prec: E::NAME,
        grad,
        label,
        isa,
        order,
        block,
        flavour,
        rsqrt,
        model_ops,
        lanes,
        pairs_per_s,
        acc,
        passed,
        same_as_gathered,
    }
}

/// Runs every variant of precision `E` on the cells, potential only and with
/// gradients. `filter` selects the prototypes run on a cell.
pub fn run_cells<E: Elem + Gk + Prototypes>(
    cells: &[Cell],
    filter: &dyn Fn(&Proto<E>, Cell) -> bool,
    gk_backend: &str,
) -> Vec<Row> {
    let protos: Vec<Proto<E>> = E::prototypes();
    let mut rows = Vec::new();
    for &cell in cells {
        let mut pool: Vec<Set<E>> = match cell {
            Cell::W1(n, _) => w1(n, 0x00F1_0001),
            Cell::W2(n, eq) => vec![w2(n, eq, 0x00F1_0002)],
        };
        let checked = CHECKED_SETS.min(pool.len());
        let oracles: Vec<Oracle> = pool[..checked].iter().map(Oracle::new).collect();
        for grad in [false, true] {
            eprintln!(
                "  {} {} {}",
                E::NAME,
                cell.label(),
                if grad { "∇" } else { "φ" }
            );
            let mut variants = vec![Variant::Reference, Variant::Green];
            variants.extend(
                protos
                    .iter()
                    .filter(|p| p.grad == grad && filter(p, cell))
                    .map(|&p| Variant::Proto(p)),
            );
            let mut reference = None;
            for v in &variants {
                let row = run_variant(v, cell, &mut pool, &oracles, grad, gk_backend, reference);
                if let Variant::Reference = v {
                    reference = Some(row.acc);
                }
                rows.push(row);
            }
        }
    }
    rows
}
