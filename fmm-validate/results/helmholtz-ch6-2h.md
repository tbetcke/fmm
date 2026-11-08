# Helmholtz CH6.2h: D8, the exponential M2L's level rule

Helmholtz task CH6.2h ([docs/helmholtz/tasks/CH6.2h.md](../../docs/helmholtz/tasks/CH6.2h.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2, §12 D8, §2 (constraints 1, 2, 7, 9), §5.3, §6.4. It proposes D8: the rule by which
`HelmholtzM2l::Auto` takes the exponential M2L ("PW") per level, measured on CH6.2f's level call
([helmholtz-ch6-2f.md](helmholtz-ch6-2f.md)). Measured on:

- **the M6** (Apple M6, `tools/m6/machine.md`): one rank × 6 threads, the matrix,
  2026-11-08 21:39–21:45, with the user's go-ahead, under the run script's quiet gate (other
  processes 16–50 % of a core before a launch, at most 86 % during one).
- **Kathleen** (two Xeon Gold 6248 sockets, 2 × 20 Cascade Lake cores, AVX-512,
  `tools/kathleen/`): job 630761 (node-c11a-002, `singlenode`, exclusive, 22:12–22:22), the
  matrix at one rank × 40 threads; job 630762 (node-c11a-003, 22:12–23:23), the large cases at
  1 × 40 and 4 × 10 (`--map-by slot:PE=10 --bind-to core`, Open MPI 4.1.6). Both with
  `auto-pw:16` added (§4).
- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores): pending (§5). A first pass ran beside
  another session's CUDA compile and is a record only.

Source 4d44501 for every run (the later commits change the tables script and docs only),
release builds, rustc 1.99.0, BLAS threads 1, the default waves budget (8 GiB per node), the
default split of the merge and shift (CH6.2f's `Auto`). Commands: §11.

## Summary

- **The rule** (§1), behind `auto_plane_wave(true)` until D8's sign-off: on each level l ≥ 2,
  PW where p_l ≥ d_pw, CH6.1e's precision rule gives a quadrature (ε ≥ 10⁻⁶) and
  Im κ_l < 5π (f64 only); Batched where p_l ≤ d_b within the dense budget; Rotation
  otherwise. d_pw is d_b + 1 (PW above Batched's crossover), except AVX-512 with two planes,
  where it is 12. The inputs are the level's degree and width, k, ε and the machine class.
  The rule has no boxes term (below).
- **Against today's `Auto`** (Batched to d_b, Rotation above), M2L summed over the levels:
  - At 10⁻⁶: 0.14–0.70× on the M6. On Kathleen 0.03–0.55× on the matrix and 0.23–0.49× on
    the large cases. The evaluation drops to 0.21–0.91× on the M6, 0.05–0.66× and 0.28–0.52×
    on Kathleen.
  - At 10⁻³: 0.47–1.07× on the M6 and 0.26–1.19× on Kathleen. The cases above 1.0 take every
    level as today's `Auto` does: the same checksum, within the ABBA spread.
- **Against the per-level best** (each level's fastest of Rotation, Batched and PW):
  - Within 0.93–1.09× on the M6 and 0.92–1.18× on Kathleen, except five ms-scale cases at
    10⁻³: on the M6 W1 and W3 at kD = 10 (1.96×, 1.99×); on Kathleen W1 at kD = 10 and 30
    (8.8×, 1.4×) and W3 at kD = 10 (6.4×).
  - Two 2–3 ms cases at 10⁻⁶ on Kathleen also exceed it (W1 and W3 at kD = 10, 1.51× and
    1.27×). There the rule ran PlaneWave's methods level for level, so the excess is
    run-to-run spread.
  - All of the excess is on levels with few boxes (§6), and most of it is Batched at p = 7
    on level 2, which today's `Auto` chooses too.
- **No boxes term for PW** (§6). On CH6.2f's level call, PW loses to Rotation only at 10⁻³ on
  levels 2–3: at most 2.2× and 3.7 ms per evaluation, at most 23 % of an 8 ms case and
  under 6 % elsewhere. At 10⁻⁶ it is faster than Rotation on every level but one (Kathleen,
  W4 clusters at kD = 1, level 2: 1.06×, 0.2 ms).
- **Batched's few-box levels** are a separate miss, of today's `Auto` (CH3.2d, the sparse-level
  term left to CH4.0). On level 2 (64 boxes, 3,096 V pairs) at p = 7, Batched takes 2.1–3.8×
  Rotation's time on the M6 and 8.5–9.3× on Kathleen (about 2 and 4 ms). Today's `Auto`
  takes p = 15 to Batched on Kathleen at 0.09–0.12 s per level against Rotation's 3–5 ms (up
  to 38×); the rule removes that by taking PW from p = 12. Proposed as a follow-up (§9).
- **d_pw on AVX-512** (§4): 12, not d_b + 1 = 16. With 16, levels at p = 12–15 run Batched.
  That is never faster beyond noise (0.91–1.03× on the cases where it changes nothing), and
  it takes 1.6–35× the M2L at 10⁻⁶ (1.29–3.88× on the large cases) and up to 4.1× at 10⁻³.
- **Accuracy** (§7): the rule's errors are 0.916–1.063× Rotation's (φ and ∇φ) on every case
  and machine, within CH6.1e's 1.1× bound. At 10⁻⁶ its checksum equals PlaneWave's on every
  case: each level that has waves runs the same call.
- **D8 proposed** (§8): `Auto` takes this rule by default, the builder's `auto_plane_wave` on.

## 1. The rule

`HelmholtzOperator::with_auto_plane_wave(bool)` and `HelmholtzFmmBuilder::auto_plane_wave(bool)`
(off by default until D8's sign-off) set the rule under `HelmholtzM2l::Auto`. The rule only
chooses which level call each level runs; the calls themselves are unchanged. For each level
l ≥ 2, with degree p_l and κ_l = k w_l (the level's width in wavelengths):

1. **PW** (`m2l_on` = PlaneWave at CH6.1e's precision) where all of these hold:
   - p_l ≥ d_pw;
   - `plane_wave::merge::select` gives the level a quadrature (iprec 2 at ε ≥ 10⁻⁶, none below);
   - Im κ_l < 5π (`AUTO_PLANE_WAVE_MAX_IM_KAPPA`; the domains CH6.2g's guard test covers);
   - the scalar is f64.
2. **Batched** where p_l ≤ d_b (`auto_batched_max_degree`, CH3.2d) and the tables fit the dense
   budget. Levels are charged from the finest up, and PW's levels are left out.
3. **Rotation** otherwise.

d_pw is `auto_plane_wave_min_degree(isa, planes)`: d_b + 1 (NEON 11 with two planes and 13
with one, AVX2 12, scalar 7 and 9), and 12 for AVX-512 with two planes, against its d_b of 15.
`with_m2l_plane_wave_min_degree` and the builder's `m2l_plane_wave_min_degree` name another
value; the measurements use this. The builder agrees d_pw over the ranks with an all-reduce
min, as for d_b, and `HelmholtzFmm::m2l_plane_wave_min_degree` reports it.

The constraints of proposal §2:
- **Constraint 9**: the inputs are k, the level's width and degree, ε and the machine class.
- **Constraint 2**: each level runs Batched's, PlaneWave's or Rotation's deterministic call
  with CH6.2f's split, so the output is the same at any thread or rank count.
- **Constraint 7**: the waves buffer stays within the builder's node budget.
- **Constraint 1**: no Laplace path is touched.

## 2. Method

The run script `tools/helmholtz/ch6-2h-runs.sh` uses CH6.2b's cases:
- the matrix: W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4 at N = 10⁵, kD = 1 and 100;
- the large cases: W1 at kD = 500, W3 at kD = 100;
- each at ε = 10⁻³ and 10⁻⁶.

Each case is one launch of `helmholtz_accuracy --m2l rotation,auto,auto-pw,planewave`, with
Kathleen's `auto-pw:16` appended, then one launch with the order reversed (ABBA). Both use
`--evaluations 2 --levels`, which times the M2L of each level in the second evaluation.
`tools/helmholtz/ch6-2h-tables.py` reads the outputs:
- each strategy's time per level, the mean of the two launches;
- each level's fastest method, each method timed by the first strategy that ran it;
- the rule's time over the fastest;
- per case, the sums, the evaluations, the largest ABBA spread, the rule's errors over
  Rotation's, and which strategies share its checksum.

Its output is the appendix.

## 3. The M6

| ε | M2L Σ auto-pw ÷ auto | evaluate auto-pw ÷ auto | auto-pw ÷ per-level best |
| --- | --- | --- | --- |
| 10⁻⁶ | 0.14–0.70 (W1 kD = 10, 30, 100: 0.51, 0.53, 0.39; W3 kD = 10, 30: 0.70, 0.21; W4: 0.14–0.36) | 0.21–0.91 | 0.93–1.04 |
| 10⁻³ | 0.47–1.07 (W1 kD = 100 0.58, W4 kD = 100 0.47–0.93; all-Batched cases 0.94–1.07) | 0.55–1.03 | 1.00–1.09; W1, W3 kD = 10: 1.96, 1.99 |

- **At 10⁻⁶** the rule takes PW on every level of every case (every level's degree is at
  least d_pw = 11), with the checksum and time of PlaneWave.
- **At 10⁻³** PW takes the levels at p ≥ 11 and Batched the rest. Where every level is at
  p ≤ 10 (the W4 at kD = 1, W1 and W3 at kD = 10, W1 at kD = 30 level 3), the rule is today's
  `Auto`: same checksum, ratio 0.94–1.07 against an ABBA spread of up to 1.44.
- **Above 1.10× a level's fastest**, nine levels (§6):
  - p = 7 Batched on level 2: 2.1–3.8× Rotation, 1.1–2.2 ms;
  - W3 at kD = 10 level 3: 1.18×, 0.3 ms;
  - p = 12 PW on level 2 at 10⁻³ (W1, W3 at kD = 30): 1.19–1.25× Rotation, 0.5–0.6 ms.

## 4. Kathleen, and d_pw on AVX-512

| ε | cases | M2L Σ auto-pw ÷ auto | evaluate auto-pw ÷ auto | auto-pw ÷ per-level best |
| --- | --- | --- | --- | --- |
| 10⁻⁶ | matrix | 0.03–0.55 | 0.05–0.66 | 0.96–1.03; W3, W1 kD = 10: 1.27, 1.51 (2–3 ms, PlaneWave's methods) |
| 10⁻⁶ | large, 1 × 40 and 4 × 10 | 0.23–0.49 | 0.28–0.52 | 0.99–1.01 |
| 10⁻³ | matrix | 0.26–1.19 | 0.35–1.04 | 1.01–1.18; W1 kD = 10, 30: 8.8, 1.4; W3 kD = 10: 6.4 |
| 10⁻³ | large | 0.64–0.99 | 0.65–0.99 | 0.92–1.02 |

Today's `Auto` is slow on Kathleen's small cases at 10⁻⁶. It takes p ≤ 15 to Batched, and
Batched on a level of few boxes costs a fixed tenth of a second:
- W1 at kD = 10, level 2 (p = 15, 3,096 pairs): 0.114 s against Rotation's 0.003 s;
- W3 at kD = 10, levels 2–3: 0.12 and 0.09 s;
- W1 at kD = 30, level 2 (p = 12): 0.034 s, against PW's 3.2 ms and Rotation's 2.1 ms.

The rule takes these levels to PW. Hence the matrix ratios of 0.03–0.05 at kD = 10 and 0.23
at kD = 30.

**d_pw: 12 or d_b + 1 = 16.** The column auto-pw:16 runs the rule with d_pw = 16, which
leaves p = 12–15 to Batched. Its M2L over the rule's (d_pw = 12):

| | matrix 10⁻⁶ | matrix 10⁻³ | large 10⁻⁶ | large 10⁻³ |
| --- | --- | --- | --- | --- |
| auto-pw:16 ÷ auto-pw | 1.62–34.7 | 0.91–4.09 | 1.29–3.88 | 1.00–1.12 |

- Where the two choose the same methods (the W4 at kD = 1, 10⁻³), the ratio is 0.91–1.03,
  which is noise.
- Wherever they differ, 16 is slower: on fine levels at p = 15 (W3 at kD = 100 levels 4–6,
  10⁻⁶: Batched 24.9 s against PW 6.27 s on level 6), and on coarse levels at p = 12–15
  (above).
- CH6.2b's table (plan) had Batched faster than PW at p ≤ 11 on AVX-512. p = 12 occurs here
  only on level 2, where PW wins over Batched 11×. 12 is kept.
- The AVX2 machine class (d_pw = 12, d_b + 1) and the scalar classes remain unmeasured: no
  AVX2 node of our own (CH6.2b's report).

## 5. locust (pending)

The run is the matrix and the large cases at one rank × 72 threads behind a strict gate:
- before the run, other processes below 30 % of a core for five minutes, excluding another
  user's long-running Python process (about one core, accepted in CH6.2f);
- during the run, a sample every 7 s.

Output goes to `/data/ucahtbe/logs/ch6-2h-b`. As of this writing it waits for another
session's GPU sweep (`degree_sweep`) to finish.

The first pass (`bench-results/ch6-2h/locust-first`, 20:36–20:48) ran beside another
session's CUDA compile, which a looser gate let through. It is kept as a record only; its
summary is in the appendix. At 10⁻⁶ it agrees with the M6 (auto-pw ÷ auto 0.21–1.07, ÷ best
0.91–1.07). At 10⁻³ it shows Batched at p = 7 well behind Rotation at 72 threads, not only on
few-box levels: 23–28× on level 2, 1.1–7.5× on levels of up to 3.5·10⁵ pairs. The clean pass
decides whether that holds.

## 6. Levels of few boxes

The roadmap asks whether CH6.2f's split brings coarse levels within the rule or needs a boxes
term (or Rotation on few-box levels). The rule's levels above 1.10× the level's fastest, on
the M6 and Kathleen (appendix, last tables):

| method the rule ran | where | ratio to the fastest (Rotation) | lost per evaluation |
| --- | --- | --- | --- |
| PW, 10⁻³ | level 2 at p = 12 (W1, W3 at kD = 30); Kathleen also W4 clusters kD = 100 levels 2–3 (p = 21, 13), W3 kD = 100 level 2 (p = 30) | M6 1.19–1.25; Kathleen 1.15–2.16 | 0.5–3.7 ms; ≤ 23 % of an 8 ms case, ≤ 6 % elsewhere |
| PW, 10⁻⁶ | none above 1.10×; one level at 1.06× (Kathleen, W4 clusters kD = 1, level 2) | – | – |
| Batched p = 7–8, 10⁻³ | level 2 (64 boxes) of every case at p = 7; level 3 of W3 kD = 10, W4 clusters; W1 and W3 kD = 30 level 3 (p = 8); W4 Plummer level 8 (35,148 pairs) | M6 1.18–3.79; Kathleen 1.14–9.31 | 0.3–4.4 ms |

**PW needs no boxes term.**
- After CH6.2f's split, PW on levels 2–3 is within 1.25× of Rotation on the M6. On Kathleen
  it is within 2.2× at 10⁻³, at most 3.7 ms per evaluation.
- At 10⁻³, PW runs iprec 2 (CH6.1e's rule), more waves than the tolerance needs. That is
  where the small loss comes from. A coarser precision at 10⁻³ is CH6.1f's question, not a
  boxes term.
- A term reading the level's boxes would save at most 3.7 ms per evaluation on these cases.
  It would add an input that constraint 9 allows but that no measured case pays for.

**Batched on few-box levels** is today's `Auto` miss, in the same place:
- CH3.2d recorded it, and CH4.0 holds the sparse-level term.
- Per class, Batched reads the class's whole table whatever the number of pairs: 316 classes
  for 3,096 pairs on level 2.
- The cost is about 2 ms on the M6 and 4 ms on Kathleen at p = 7, and 0.1 s at p = 15 on
  Kathleen, which the rule now gives to PW.
- Fixing it means Rotation on levels whose pairs per class fall below a threshold, or a split
  of the level call as in CH6.2f. That is Batched's rule, not D8's, so it is proposed as a
  follow-up (§9).

## 7. Accuracy

- The rule's errors over Rotation's (φ / ∇φ, against the direct sum on the harness's
  points):
  - 0.969–1.063 / 0.916–1.000 at 10⁻⁶;
  - 1.000 / 1.000 at 10⁻³ (three decimals), PW levels included.
- These are the same on the M6 and Kathleen, and at 1 × 40 and 4 × 10. They are within
  CH6.1e's 1.1× of Rotation.
- Checksums:
  - at 10⁻⁶ the rule's equals PlaneWave's on every case;
  - at 10⁻³ it equals today's `Auto`'s where no level is PW;
  - on Kathleen it equals auto-pw:16's where the two choose the same methods;
  - at 4 × 10 it equals 1 × 40's.

## 8. D8 proposed

> **D8.** `HelmholtzM2l::Auto` (the default) takes the exponential M2L on a level l ≥ 2 where:
> - p_l ≥ d_pw: Batched's crossover + 1, 12 on AVX-512 with two planes;
> - CH6.1e's precision rule gives the level a quadrature (ε ≥ 10⁻⁶);
> - Im κ_l < 5π;
> - the scalar is f64.
>
> Batched takes p_l ≤ d_b on the remaining levels within the dense budget, and Rotation the
> rest.
>
> The inputs are the level's degree and κ_l, ε and the machine class. There is no boxes term:
> on CH6.2f's level call the exponential M2L's few-box levels lose at most 3.7 ms per
> evaluation at 10⁻³ and at most 0.2 ms (1.06×) at 10⁻⁶.
>
> Batched's few-box levels go to a follow-up of Batched's rule. The builder's
> `auto_plane_wave` becomes `true`.

On sign-off, on this branch:
- the builder's and the operator's defaults turn on;
- the tests that pin today's `Auto` name the setting off;
- the docs (module "M2L", `HelmholtzM2l::Auto`, the crate `CLAUDE.md`s) are restated;
- proposal §5.3 and §6.4 ("opt-in until CH6.2") are amended.

## 9. Follow-ups

- **Batched's few-box levels** (§6): a sparse-level term in `Auto`'s Batched choice (pairs per
  class, or the level's boxes) or a split of Batched's level call: a sentence added to CH4.0
  (the M2L rule's sparse-level term) in the roadmap.
- **locust's clean pass** (§5): pending on this PR.
- **d_pw on AVX2 and scalar**: d_b + 1, unmeasured; with the first AVX2 node.

## 10. Tests and checks

- `tests/helmholtz/plane_wave.rs`:
  - `auto_takes_the_waves_by_its_rule`: the methods per level for named d_b and d_pw at
    10⁻³, 10⁻⁶ and 10⁻⁸; off, and set after `with_m2l`; Rotation ignores it.
  - `auto_waves_by_the_machine_class_and_the_damping`: the ISA's d_pw through `with_p2p`;
    k = 20 + 100i gives Rotation on level 2 (Im κ ≥ 5π) and PW on level 3, while PlaneWave
    named keeps level 2; f32 gets no waves.
  - `the_plane_wave_degrees_follow_the_crossovers`.
- `tests/helmholtz_fmm.rs` (`check_auto_plane_wave`, scenarios 0 and 2; one rank and the MPI
  list's 2 and 4):
  - d_b = 0 and d_pw = 0 give PlaneWave bit for bit;
  - the split (PW, PW, PW, Batched) has errors within 1.1× Rotation's, 2 threads bit for bit,
    and P ranks equal the one-rank FMM;
  - the agreed d_pw equals the all-reduce min of `auto_plane_wave_min_degree`, None by default.
- `fmm-validate/tests/helmholtz_accuracy.rs` (`check_auto`): with waves at d_b = 0, `Auto`
  equals PlaneWave's checksum and precisions; errors within 1.1× Rotation's; off gives None.

`/verify` on the M6 (source 836b6b7; code as 4d44501):

| command | result |
| --- | --- |
| `cargo fmt --all -- --check` | passed |
| `cargo clippy -- -D warnings`, `--examples`, `--workspace --all-targets` | passed |
| `cargo doc --no-deps` | passed |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | 1339 passed (16.6 min) |
| `RUST_MIN_STACK=8388608 cargo test` (default members) | not run separately; its tests are a subset of the workspace run |
| `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks | all 12 executables passed |
| `cargo test -p nd-fmm-validate --test helmholtz_accuracy` | passed (43.6 s) |
| `tests/helmholtz_fmm.rs` one-rank debug run | 242 s (216 s at CH6.2f), above nd-fmm-exec's one-minute budget as before |

## 11. Reproducing

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
# the M6 (quiet gate built in)
THREADS=6 tools/helmholtz/ch6-2h-runs.sh bench-results/ch6-2h/m6 matrix
# Kathleen (tools/kathleen/sync.sh first): jobs 630761 and 630762
tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
  'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
  'THREADS=40 EXTRA=auto-pw:16 tools/helmholtz/ch6-2h-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2h matrix'
tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
  'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
  'THREADS=40 SPLITS="1x40 4x10" EXTRA=auto-pw:16 tools/helmholtz/ch6-2h-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2h-large large'
# locust (tools/gh200/sync.sh first; quiet node)
tools/gh200/remote.sh 'THREADS=72 SPLITS=1x72 tools/helmholtz/ch6-2h-runs.sh /data/ucahtbe/logs/ch6-2h-b matrix large'
# The tables (outputs fetched to bench-results/ch6-2h/<machine>/)
tools/helmholtz/ch6-2h-tables.py m6=bench-results/ch6-2h/m6
tools/helmholtz/ch6-2h-tables.py --extra auto-pw:16 kathleen=bench-results/ch6-2h/kathleen \
  kathleen-large=bench-results/ch6-2h/kathleen-large
```

The raw outputs: `~/projects/fmm-archive/m6/logs/ch6-2h`, `kathleen/logs/ch6-2h{,-large}`,
`locust/logs/ch6-2h-first` (copies of `bench-results/ch6-2h/`, which git ignores).

## Appendix: the tables script's output

### The M6

#### m6

##### W1 (N = 796), kD = 10, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0010 | 0.0021 (b-f) | 0.0020 (b-f) | 0.0015 | rot. | 2.07 |
| Σ | | | 0.0010 | 0.0021 | 0.0020 | 0.0015 | best 0.0010 | |
| evaluate | | | 0.002 | 0.003 | 0.003 | 0.002 | | |

##### W1 (N = 796), kD = 10, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0041 | 0.0047 (rot.) | 0.0024 (pw i2) | 0.0024 | pw i2 | 1.00 |
| Σ | | | 0.0041 | 0.0047 | 0.0024 | 0.0024 | best 0.0024 | |
| evaluate | | | 0.005 | 0.005 | 0.003 | 0.003 | | |

##### W1 (N = 7162), kD = 30, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0023 | 0.0023 (rot.) | 0.0030 (pw i2) | 0.0029 | rot. | 1.25 |
| 3 | 8 | 32256 | 0.0089 | 0.0097 (b-f) | 0.0092 (b-f) | 0.0110 | rot. | 1.09 |
| Σ | | | 0.0112 | 0.0121 | 0.0121 | 0.0138 | best 0.0112 | |
| evaluate | | | 0.016 | 0.017 | 0.017 | 0.019 | | |

##### W1 (N = 7162), kD = 30, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0047 | 0.0047 (rot.) | 0.0038 (pw i2) | 0.0036 | pw i2 | 1.00 |
| 3 | 15 | 32256 | 0.0401 | 0.0397 (rot.) | 0.0198 (pw i2) | 0.0191 | pw i2 | 1.00 |
| Σ | | | 0.0447 | 0.0445 | 0.0236 | 0.0227 | best 0.0227 | |
| evaluate | | | 0.053 | 0.052 | 0.031 | 0.030 | | |

##### W1 (N = 79578), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0252 | 0.0286 (rot.) | 0.0181 (pw i2) | 0.0185 | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.1064 | 0.1082 (rot.) | 0.0484 (pw i2) | 0.0475 | pw i2 | 1.00 |
| 4 | 11 | 385344 | 0.2202 | 0.2191 (rot.) | 0.1285 (pw i2) | 0.1245 | pw i2 | 1.00 |
| 5 | 7 | 244944 | 0.0516 | 0.0380 (b-f) | 0.0327 (b-f) | 0.0915 | b-f | 1.00 |
| Σ | | | 0.4034 | 0.3938 | 0.2277 | 0.2819 | best 0.2284 | |
| evaluate | | | 0.531 | 0.517 | 0.356 | 0.407 | | |

##### W1 (N = 79578), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0461 | 0.0396 (rot.) | 0.0202 (pw i2) | 0.0203 | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.2171 | 0.1859 (rot.) | 0.0573 (pw i2) | 0.0577 | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.4875 | 0.5108 (rot.) | 0.1676 (pw i2) | 0.1645 | pw i2 | 1.00 |
| 5 | 15 | 244944 | 0.3123 | 0.3688 (rot.) | 0.1826 (pw i2) | 0.1816 | pw i2 | 1.00 |
| Σ | | | 1.0630 | 1.1051 | 0.4277 | 0.4242 | best 0.4242 | |
| evaluate | | | 1.366 | 1.408 | 0.731 | 0.727 | | |

##### W3 (N = 108849), kD = 30, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0025 | 0.0023 (rot.) | 0.0029 (pw i2) | 0.0029 | rot. | 1.19 |
| 3 | 8 | 53352 | 0.0152 | 0.0133 (b-f) | 0.0134 (b-f) | 0.0143 | b-f | 1.00 |
| 4 | 7 | 584136 | 0.1306 | 0.0771 (b-f) | 0.0765 (b-f) | 0.0867 | b-f | 1.00 |
| Σ | | | 0.1483 | 0.0927 | 0.0927 | 0.1040 | best 0.0929 | |
| evaluate | | | 0.224 | 0.169 | 0.172 | 0.180 | | |

##### W3 (N = 108849), kD = 30, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0061 | 0.0050 (rot.) | 0.0038 (pw i2) | 0.0038 | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0728 | 0.0830 (rot.) | 0.0237 (pw i2) | 0.0239 | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.9150 | 0.7737 (rot.) | 0.1557 (pw i2) | 0.1545 | pw i2 | 1.00 |
| Σ | | | 0.9940 | 0.8617 | 0.1832 | 0.1823 | best 0.1823 | |
| evaluate | | | 1.109 | 0.977 | 0.298 | 0.297 | | |

##### W3 (N = 4032), kD = 10, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0007 | 0.0026 (b-f) | 0.0026 (b-f) | 0.0015 | rot. | 3.56 |
| 3 | 7 | 5544 | 0.0014 | 0.0017 (b-f) | 0.0018 (b-f) | 0.0035 | rot. | 1.18 |
| Σ | | | 0.0022 | 0.0043 | 0.0044 | 0.0050 | best 0.0022 | |
| evaluate | | | 0.014 | 0.016 | 0.016 | 0.017 | | |

##### W3 (N = 4032), kD = 10, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0041 | 0.0044 (rot.) | 0.0024 (pw i2) | 0.0024 | pw i2 | 1.00 |
| 3 | 15 | 5544 | 0.0086 | 0.0084 (rot.) | 0.0066 (pw i2) | 0.0072 | pw i2 | 1.00 |
| Σ | | | 0.0128 | 0.0129 | 0.0090 | 0.0096 | best 0.0096 | |
| evaluate | | | 0.044 | 0.045 | 0.041 | 0.041 | | |

##### W4 clusters (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0030 (b-f) | 0.0029 (b-f) | 0.0013 | rot. | 3.69 |
| 3 | 7 | 19782 | 0.0047 | 0.0038 (b-f) | 0.0040 (b-f) | 0.0054 | b-f | 1.00 |
| 4 | 7 | 43392 | 0.0106 | 0.0077 (b-f) | 0.0079 (b-f) | 0.0092 | b-f | 1.00 |
| 5 | 7 | 106500 | 0.0346 | 0.0157 (b-f) | 0.0164 (b-f) | 0.0201 | b-f | 1.00 |
| 6 | 7 | 214818 | 0.0535 | 0.0359 (b-f) | 0.0311 (b-f) | 0.0388 | b-f | 1.00 |
| 7 | 7 | 289194 | 0.0722 | 0.0458 (b-f) | 0.0450 (b-f) | 0.0489 | b-f | 1.00 |
| 8 | 7 | 351936 | 0.0906 | 0.0540 (b-f) | 0.0597 (b-f) | 0.0618 | b-f | 1.00 |
| Σ | | | 0.2669 | 0.1660 | 0.1669 | 0.1855 | best 0.1638 | |
| evaluate | | | 0.529 | 0.409 | 0.419 | 0.434 | | |

##### W4 clusters (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0046 | 0.0050 (rot.) | 0.0022 (pw i2) | 0.0023 | pw i2 | 1.00 |
| 3 | 15 | 19782 | 0.0390 | 0.0383 (rot.) | 0.0102 (pw i2) | 0.0102 | pw i2 | 1.00 |
| 4 | 15 | 43392 | 0.0909 | 0.0735 (rot.) | 0.0183 (pw i2) | 0.0192 | pw i2 | 1.00 |
| 5 | 15 | 106500 | 0.1805 | 0.1587 (rot.) | 0.0390 (pw i2) | 0.0392 | pw i2 | 1.00 |
| 6 | 15 | 214818 | 0.3239 | 0.3221 (rot.) | 0.0723 (pw i2) | 0.0722 | pw i2 | 1.00 |
| 7 | 15 | 289194 | 0.5968 | 0.4326 (rot.) | 0.0962 (pw i2) | 0.0949 | pw i2 | 1.00 |
| 8 | 15 | 351936 | 0.5298 | 0.5293 (rot.) | 0.1188 (pw i2) | 0.1147 | pw i2 | 1.00 |
| Σ | | | 1.7655 | 1.5596 | 0.3570 | 0.3527 | best 0.3527 | |
| evaluate | | | 2.472 | 2.299 | 1.056 | 1.048 | | |

##### W4 clusters (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 21 | 3096 | 0.0116 | 0.0120 (rot.) | 0.0082 (pw i2) | 0.0082 | pw i2 | 1.00 |
| 3 | 13 | 19782 | 0.0276 | 0.0309 (rot.) | 0.0153 (pw i2) | 0.0156 | pw i2 | 1.00 |
| 4 | 8 | 43392 | 0.0166 | 0.0125 (b-f) | 0.0128 (b-f) | 0.0155 | b-f | 1.00 |
| 5 | 7 | 106500 | 0.0300 | 0.0166 (b-f) | 0.0170 (b-f) | 0.0245 | b-f | 1.00 |
| 6 | 7 | 214818 | 0.0560 | 0.0333 (b-f) | 0.0343 (b-f) | 0.0426 | b-f | 1.00 |
| 7 | 7 | 289194 | 0.0737 | 0.0455 (b-f) | 0.0477 (b-f) | 0.0551 | b-f | 1.00 |
| 8 | 7 | 351936 | 0.0887 | 0.0538 (b-f) | 0.0554 (b-f) | 0.0646 | b-f | 1.00 |
| Σ | | | 0.3042 | 0.2048 | 0.1908 | 0.2262 | best 0.1857 | |
| evaluate | | | 0.589 | 0.465 | 0.447 | 0.480 | | |

##### W4 clusters (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 25 | 3096 | 0.0217 | 0.0204 (rot.) | 0.0098 (pw i2) | 0.0096 | pw i2 | 1.00 |
| 3 | 16 | 19782 | 0.0490 | 0.0513 (rot.) | 0.0187 (pw i2) | 0.0207 | pw i2 | 1.00 |
| 4 | 15 | 43392 | 0.0753 | 0.0663 (rot.) | 0.0279 (pw i2) | 0.0277 | pw i2 | 1.00 |
| 5 | 15 | 106500 | 0.1616 | 0.1624 (rot.) | 0.0447 (pw i2) | 0.0443 | pw i2 | 1.00 |
| 6 | 15 | 214818 | 0.3249 | 0.3402 (rot.) | 0.0806 (pw i2) | 0.0813 | pw i2 | 1.00 |
| 7 | 15 | 289194 | 0.4403 | 0.4383 (rot.) | 0.1031 (pw i2) | 0.1040 | pw i2 | 1.00 |
| 8 | 15 | 351936 | 0.5364 | 0.5332 (rot.) | 0.1193 (pw i2) | 0.1179 | pw i2 | 1.00 |
| Σ | | | 1.6091 | 1.6120 | 0.4041 | 0.4054 | best 0.4054 | |
| evaluate | | | 2.337 | 2.347 | 1.108 | 1.119 | | |

##### W4 cube (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0007 | 0.0028 (b-f) | 0.0028 (b-f) | 0.0014 | rot. | 3.79 |
| 3 | 7 | 53352 | 0.0133 | 0.0083 (b-f) | 0.0092 (b-f) | 0.0094 | b-f | 1.00 |
| 4 | 7 | 584136 | 0.1335 | 0.0803 (b-f) | 0.0777 (b-f) | 0.0832 | b-f | 1.00 |
| Σ | | | 0.1475 | 0.0914 | 0.0896 | 0.0940 | best 0.0893 | |
| evaluate | | | 0.214 | 0.155 | 0.154 | 0.159 | | |

##### W4 cube (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0042 | 0.0043 (rot.) | 0.0021 (pw i2) | 0.0021 | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0752 | 0.0811 (rot.) | 0.0179 (pw i2) | 0.0181 | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.9510 | 1.0882 (rot.) | 0.1492 (pw i2) | 0.1490 | pw i2 | 1.00 |
| Σ | | | 1.0304 | 1.1737 | 0.1692 | 0.1692 | best 0.1692 | |
| evaluate | | | 1.132 | 1.275 | 0.273 | 0.271 | | |

##### W4 cube (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0316 | 0.0316 (rot.) | 0.0190 (pw i2) | 0.0173 | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.1183 | 0.1393 (rot.) | 0.0509 (pw i2) | 0.0506 | pw i2 | 1.00 |
| 4 | 11 | 584136 | 0.3806 | 0.3736 (rot.) | 0.1834 (pw i2) | 0.1796 | pw i2 | 1.00 |
| Σ | | | 0.5304 | 0.5444 | 0.2533 | 0.2476 | best 0.2476 | |
| evaluate | | | 0.636 | 0.649 | 0.354 | 0.347 | | |

##### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0424 | 0.0425 (rot.) | 0.0206 (pw i2) | 0.0207 | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.2473 | 0.2104 (rot.) | 0.0592 (pw i2) | 0.0602 | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.8245 | 0.9653 (rot.) | 0.2342 (pw i2) | 0.2337 | pw i2 | 1.00 |
| Σ | | | 1.1142 | 1.2182 | 0.3141 | 0.3145 | best 0.3145 | |
| evaluate | | | 1.250 | 1.355 | 0.443 | 0.447 | | |

##### W4 plummer (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0029 (b-f) | 0.0029 (b-f) | 0.0013 | rot. | 3.72 |
| 3 | 7 | 53352 | 0.0121 | 0.0090 (b-f) | 0.0087 (b-f) | 0.0100 | b-f | 1.00 |
| 4 | 7 | 53352 | 0.0132 | 0.0087 (b-f) | 0.0088 (b-f) | 0.0104 | b-f | 1.00 |
| 5 | 7 | 109548 | 0.0277 | 0.0173 (b-f) | 0.0185 (b-f) | 0.0180 | b-f | 1.00 |
| 6 | 7 | 205458 | 0.0535 | 0.0296 (b-f) | 0.0294 (b-f) | 0.0316 | b-f | 1.00 |
| 7 | 7 | 333330 | 0.0838 | 0.0487 (b-f) | 0.0545 (b-f) | 0.0485 | pw i2 | 1.00 |
| 8 | 7 | 35148 | 0.0097 | 0.0065 (b-f) | 0.0066 (b-f) | 0.0080 | b-f | 1.00 |
| Σ | | | 0.2008 | 0.1227 | 0.1293 | 0.1277 | best 0.1204 | |
| evaluate | | | 0.454 | 0.362 | 0.363 | 0.364 | | |

##### W4 plummer (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0048 | 0.0043 (rot.) | 0.0023 (pw i2) | 0.0023 | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0907 | 0.0922 (rot.) | 0.0186 (pw i2) | 0.0186 | pw i2 | 1.00 |
| 4 | 15 | 53352 | 0.1091 | 0.0807 (rot.) | 0.0213 (pw i2) | 0.0192 | pw i2 | 1.00 |
| 5 | 15 | 109548 | 0.1937 | 0.1632 (rot.) | 0.0336 (pw i2) | 0.0345 | pw i2 | 1.00 |
| 6 | 15 | 205458 | 0.3643 | 0.3584 (rot.) | 0.0594 (pw i2) | 0.0595 | pw i2 | 1.00 |
| 7 | 15 | 333330 | 0.4952 | 0.4945 (rot.) | 0.0890 (pw i2) | 0.0900 | pw i2 | 1.00 |
| 8 | 15 | 35148 | 0.0535 | 0.0527 (rot.) | 0.0165 (pw i2) | 0.0175 | pw i2 | 1.00 |
| Σ | | | 1.3113 | 1.2460 | 0.2405 | 0.2417 | best 0.2417 | |
| evaluate | | | 1.986 | 1.920 | 0.874 | 0.875 | | |

##### W4 plummer (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0299 | 0.0367 (rot.) | 0.0186 (pw i2) | 0.0194 | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.1304 | 0.1323 (rot.) | 0.0535 (pw i2) | 0.0509 | pw i2 | 1.00 |
| 4 | 11 | 380178 | 0.3051 | 0.2591 (rot.) | 0.1329 (pw i2) | 0.1340 | pw i2 | 1.00 |
| 5 | 7 | 109548 | 0.0313 | 0.0192 (b-f) | 0.0187 (b-f) | 0.0239 | b-f | 1.00 |
| 6 | 7 | 205458 | 0.0522 | 0.0317 (b-f) | 0.0346 (b-f) | 0.0345 | b-f | 1.00 |
| 7 | 7 | 333330 | 0.0830 | 0.0495 (b-f) | 0.0560 (b-f) | 0.0547 | b-f | 1.00 |
| 8 | 7 | 35148 | 0.0100 | 0.0073 (b-f) | 0.0070 (b-f) | 0.0082 | b-f | 1.00 |
| Σ | | | 0.6419 | 0.5358 | 0.3212 | 0.3256 | best 0.3121 | |
| evaluate | | | 0.907 | 0.797 | 0.573 | 0.578 | | |

##### W4 plummer (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0473 | 0.0476 (rot.) | 0.0217 (pw i2) | 0.0216 | pw i2 | 1.00 |
| 3 | 21 | 53352 | 0.1975 | 0.1988 (rot.) | 0.0607 (pw i2) | 0.0605 | pw i2 | 1.00 |
| 4 | 15 | 380178 | 0.5682 | 0.5701 (rot.) | 0.1718 (pw i2) | 0.1743 | pw i2 | 1.00 |
| 5 | 15 | 109548 | 0.1643 | 0.1640 (rot.) | 0.0436 (pw i2) | 0.0445 | pw i2 | 1.00 |
| 6 | 15 | 205458 | 0.4182 | 0.3624 (rot.) | 0.0655 (pw i2) | 0.0649 | pw i2 | 1.00 |
| 7 | 15 | 333330 | 0.5017 | 0.5932 (rot.) | 0.0992 (pw i2) | 0.0995 | pw i2 | 1.00 |
| 8 | 15 | 35148 | 0.0589 | 0.0558 (rot.) | 0.0165 (pw i2) | 0.0167 | pw i2 | 1.00 |
| Σ | | | 1.9559 | 1.9920 | 0.4790 | 0.4820 | best 0.4820 | |
| evaluate | | | 2.628 | 2.660 | 1.119 | 1.113 | | |

##### W4 sphere (N = 100000), kD = 1, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0008 | 0.0029 (b-f) | 0.0029 (b-f) | 0.0013 | rot. | 3.70 |
| 3 | 7 | 53352 | 0.0122 | 0.0085 (b-f) | 0.0084 (b-f) | 0.0096 | b-f | 1.00 |
| 4 | 7 | 385344 | 0.0954 | 0.0593 (b-f) | 0.0538 (b-f) | 0.0632 | b-f | 1.00 |
| 5 | 7 | 325404 | 0.0755 | 0.0460 (b-f) | 0.0595 (b-f) | 0.0924 | b-f | 1.00 |
| Σ | | | 0.1839 | 0.1168 | 0.1246 | 0.1665 | best 0.1147 | |
| evaluate | | | 0.301 | 0.228 | 0.233 | 0.277 | | |

##### W4 sphere (N = 100000), kD = 1, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0043 | 0.0048 (rot.) | 0.0023 (pw i2) | 0.0022 | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0980 | 0.0857 (rot.) | 0.0189 (pw i2) | 0.0189 | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.5505 | 0.5507 (rot.) | 0.1223 (pw i2) | 0.1168 | pw i2 | 1.00 |
| 5 | 15 | 325404 | 0.4694 | 0.4648 (rot.) | 0.1822 (pw i2) | 0.1834 | pw i2 | 1.00 |
| Σ | | | 1.1221 | 1.1061 | 0.3258 | 0.3213 | best 0.3213 | |
| evaluate | | | 1.426 | 1.408 | 0.619 | 0.622 | | |

##### W4 sphere (N = 100000), kD = 100, ε = 1e-3, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0287 | 0.0276 (rot.) | 0.0179 (pw i2) | 0.0191 | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.1694 | 0.1279 (rot.) | 0.0530 (pw i2) | 0.0514 | pw i2 | 1.00 |
| 4 | 11 | 385344 | 0.2547 | 0.2555 (rot.) | 0.1422 (pw i2) | 0.1372 | pw i2 | 1.00 |
| 5 | 7 | 325404 | 0.0789 | 0.0491 (b-f) | 0.0472 (b-f) | 0.1284 | b-f | 1.00 |
| Σ | | | 0.5318 | 0.4601 | 0.2603 | 0.3360 | best 0.2568 | |
| evaluate | | | 0.677 | 0.609 | 0.398 | 0.475 | | |

##### W4 sphere (N = 100000), kD = 100, ε = 1e-6, 1 × 6

| level | p | V pairs | rotation | auto | auto-pw | planewave | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0441 | 0.0438 (rot.) | 0.0214 (pw i2) | 0.0210 | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.2536 | 0.2179 (rot.) | 0.0621 (pw i2) | 0.0615 | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.5688 | 0.6668 (rot.) | 0.1865 (pw i2) | 0.1865 | pw i2 | 1.00 |
| 5 | 15 | 325404 | 0.4762 | 0.4759 (rot.) | 0.2379 (pw i2) | 0.2322 | pw i2 | 1.00 |
| Σ | | | 1.3427 | 1.4044 | 0.5079 | 0.5012 | best 0.5012 | |
| evaluate | | | 1.677 | 1.739 | 0.842 | 0.833 | | |

#### Summary

| machine | workload | kD | ε | split | M2L Σ rotation | auto | auto-pw | planewave | best per level | auto-pw/best | auto-pw/auto | evaluate rotation | auto | auto-pw | planewave | auto-pw/auto | ABBA max | auto-pw ÷ rot. error φ/∇φ | auto-pw's checksum also |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| m6 | W1 | 10 | 1e-3 | 1×6 | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 | 1.96 | 0.94 | 0.002 | 0.003 | 0.003 | 0.002 | 0.95 | 1.12 | 1.000/1.000 | auto |
| m6 | W1 | 10 | 1e-6 | 1×6 | 0.004 | 0.005 | 0.002 | 0.002 | 0.002 | 1.01 | 0.51 | 0.005 | 0.005 | 0.003 | 0.003 | 0.59 | 1.03 | 1.005/0.995 | planewave |
| m6 | W1 | 30 | 1e-3 | 1×6 | 0.011 | 0.012 | 0.012 | 0.014 | 0.011 | 1.08 | 1.01 | 0.016 | 0.017 | 0.017 | 0.019 | 1.03 | 1.02 | 1.000/1.000 | – |
| m6 | W1 | 30 | 1e-6 | 1×6 | 0.045 | 0.044 | 0.024 | 0.023 | 0.023 | 1.04 | 0.53 | 0.053 | 0.052 | 0.031 | 0.030 | 0.60 | 1.09 | 1.000/1.000 | planewave |
| m6 | W1 | 100 | 1e-3 | 1×6 | 0.403 | 0.394 | 0.228 | 0.282 | 0.228 | 1.00 | 0.58 | 0.531 | 0.517 | 0.356 | 0.407 | 0.69 | 1.31 | 1.000/1.000 | – |
| m6 | W1 | 100 | 1e-6 | 1×6 | 1.063 | 1.105 | 0.428 | 0.424 | 0.424 | 1.01 | 0.39 | 1.366 | 1.408 | 0.731 | 0.727 | 0.52 | 1.44 | 0.995/0.980 | planewave |
| m6 | W3 (N = 108849) | 30 | 1e-3 | 1×6 | 0.148 | 0.093 | 0.093 | 0.104 | 0.093 | 1.00 | 1.00 | 0.224 | 0.169 | 0.172 | 0.180 | 1.02 | 1.12 | 1.000/1.000 | – |
| m6 | W3 (N = 108849) | 30 | 1e-6 | 1×6 | 0.994 | 0.862 | 0.183 | 0.182 | 0.182 | 1.01 | 0.21 | 1.109 | 0.977 | 0.298 | 0.297 | 0.31 | 1.37 | 0.973/0.964 | planewave |
| m6 | W3 (N = 4032) | 10 | 1e-3 | 1×6 | 0.002 | 0.004 | 0.004 | 0.005 | 0.002 | 1.99 | 1.01 | 0.014 | 0.016 | 0.016 | 0.017 | 0.99 | 1.27 | 1.000/1.000 | auto |
| m6 | W3 (N = 4032) | 10 | 1e-6 | 1×6 | 0.013 | 0.013 | 0.009 | 0.010 | 0.010 | 0.93 | 0.70 | 0.044 | 0.045 | 0.041 | 0.041 | 0.91 | 1.42 | 0.991/0.947 | planewave |
| m6 | W4 clusters (N = 100000) | 1 | 1e-3 | 1×6 | 0.267 | 0.166 | 0.167 | 0.185 | 0.164 | 1.02 | 1.01 | 0.529 | 0.409 | 0.419 | 0.434 | 1.02 | 1.32 | 1.000/1.000 | auto |
| m6 | W4 clusters (N = 100000) | 1 | 1e-6 | 1×6 | 1.766 | 1.560 | 0.357 | 0.353 | 0.353 | 1.01 | 0.23 | 2.472 | 2.299 | 1.056 | 1.048 | 0.46 | 1.26 | 0.969/0.916 | planewave |
| m6 | W4 clusters (N = 100000) | 100 | 1e-3 | 1×6 | 0.304 | 0.205 | 0.191 | 0.226 | 0.186 | 1.03 | 0.93 | 0.589 | 0.465 | 0.447 | 0.480 | 0.96 | 1.40 | 1.000/1.000 | – |
| m6 | W4 clusters (N = 100000) | 100 | 1e-6 | 1×6 | 1.609 | 1.612 | 0.404 | 0.405 | 0.405 | 1.00 | 0.25 | 2.337 | 2.347 | 1.108 | 1.119 | 0.47 | 1.28 | 0.970/0.921 | planewave |
| m6 | W4 cube (N = 100000) | 1 | 1e-3 | 1×6 | 0.148 | 0.091 | 0.090 | 0.094 | 0.089 | 1.00 | 0.98 | 0.214 | 0.155 | 0.154 | 0.159 | 0.99 | 1.28 | 1.000/1.000 | auto |
| m6 | W4 cube (N = 100000) | 1 | 1e-6 | 1×6 | 1.030 | 1.174 | 0.169 | 0.169 | 0.169 | 1.00 | 0.14 | 1.132 | 1.275 | 0.273 | 0.271 | 0.21 | 1.43 | 0.983/0.928 | planewave |
| m6 | W4 cube (N = 100000) | 100 | 1e-3 | 1×6 | 0.530 | 0.544 | 0.253 | 0.248 | 0.248 | 1.02 | 0.47 | 0.636 | 0.649 | 0.354 | 0.347 | 0.55 | 1.42 | 1.000/1.000 | planewave |
| m6 | W4 cube (N = 100000) | 100 | 1e-6 | 1×6 | 1.114 | 1.218 | 0.314 | 0.315 | 0.315 | 1.00 | 0.26 | 1.250 | 1.355 | 0.443 | 0.447 | 0.33 | 1.33 | 1.003/0.990 | planewave |
| m6 | W4 plummer (N = 100000) | 1 | 1e-3 | 1×6 | 0.201 | 0.123 | 0.129 | 0.128 | 0.120 | 1.07 | 1.05 | 0.454 | 0.362 | 0.363 | 0.364 | 1.00 | 1.35 | 1.000/1.000 | auto |
| m6 | W4 plummer (N = 100000) | 1 | 1e-6 | 1×6 | 1.311 | 1.246 | 0.241 | 0.242 | 0.242 | 1.00 | 0.19 | 1.986 | 1.920 | 0.874 | 0.875 | 0.45 | 1.42 | 1.063/0.940 | planewave |
| m6 | W4 plummer (N = 100000) | 100 | 1e-3 | 1×6 | 0.642 | 0.536 | 0.321 | 0.326 | 0.312 | 1.03 | 0.60 | 0.907 | 0.797 | 0.573 | 0.578 | 0.72 | 1.37 | 1.000/1.000 | – |
| m6 | W4 plummer (N = 100000) | 100 | 1e-6 | 1×6 | 1.956 | 1.992 | 0.479 | 0.482 | 0.482 | 0.99 | 0.24 | 2.628 | 2.660 | 1.119 | 1.113 | 0.42 | 1.40 | 1.013/0.955 | planewave |
| m6 | W4 sphere (N = 100000) | 1 | 1e-3 | 1×6 | 0.184 | 0.117 | 0.125 | 0.167 | 0.115 | 1.09 | 1.07 | 0.301 | 0.228 | 0.233 | 0.277 | 1.02 | 1.32 | 1.000/1.000 | auto |
| m6 | W4 sphere (N = 100000) | 1 | 1e-6 | 1×6 | 1.122 | 1.106 | 0.326 | 0.321 | 0.321 | 1.01 | 0.29 | 1.426 | 1.408 | 0.619 | 0.622 | 0.44 | 1.28 | 0.986/0.945 | planewave |
| m6 | W4 sphere (N = 100000) | 100 | 1e-3 | 1×6 | 0.532 | 0.460 | 0.260 | 0.336 | 0.257 | 1.01 | 0.57 | 0.677 | 0.609 | 0.398 | 0.475 | 0.65 | 1.14 | 1.000/1.000 | – |
| m6 | W4 sphere (N = 100000) | 100 | 1e-6 | 1×6 | 1.343 | 1.404 | 0.508 | 0.501 | 0.501 | 1.01 | 0.36 | 1.677 | 1.739 | 0.842 | 0.833 | 0.48 | 1.38 | 1.003/0.978 | planewave |

#### The rule's levels above 1.10× the fastest

| machine | workload | kD | ε | split | level | p | V pairs | auto-pw ran | fastest | ratio | lost s | lost ÷ case best |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| m6 | W1 | 10 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 2.07 | 0.0011 | 1.075 |
| m6 | W1 | 30 | 1e-3 | 1×6 | 2 | 12 | 3096 | pw i2 | rot. | 1.25 | 0.0006 | 0.051 |
| m6 | W3 (N = 108849) | 30 | 1e-3 | 1×6 | 2 | 12 | 3096 | pw i2 | rot. | 1.19 | 0.0005 | 0.005 |
| m6 | W3 (N = 4032) | 10 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 3.56 | 0.0019 | 0.862 |
| m6 | W3 (N = 4032) | 10 | 1e-3 | 1×6 | 3 | 7 | 5544 | b-f | rot. | 1.18 | 0.0003 | 0.121 |
| m6 | W4 clusters (N = 100000) | 1 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 3.69 | 0.0022 | 0.013 |
| m6 | W4 cube (N = 100000) | 1 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 3.79 | 0.0020 | 0.023 |
| m6 | W4 plummer (N = 100000) | 1 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 3.72 | 0.0021 | 0.018 |
| m6 | W4 sphere (N = 100000) | 1 | 1e-3 | 1×6 | 2 | 7 | 3096 | b-f | rot. | 3.70 | 0.0021 | 0.018 |

### Kathleen

#### kathleen

##### W1 (N = 796), kD = 10, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0005 | 0.0048 (b-f) | 0.0045 (b-f) | 0.0021 | 0.0046 (b-f) | rot. | 9.31 |
| Σ | | | 0.0005 | 0.0048 | 0.0045 | 0.0021 | 0.0046 | best 0.0005 | |
| evaluate | | | 0.002 | 0.007 | 0.006 | 0.004 | 0.007 | | |

##### W1 (N = 796), kD = 10, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0030 | 0.1144 (b-f) | 0.0033 (pw i2) | 0.0022 | 0.1036 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.0030 | 0.1144 | 0.0033 | 0.0022 | 0.1036 | best 0.0022 | |
| evaluate | | | 0.005 | 0.118 | 0.006 | 0.004 | 0.106 | | |

##### W1 (N = 7162), kD = 30, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0021 | 0.0344 (b-f) | 0.0032 (pw i2) | 0.0039 | 0.0363 (b-f) | rot. | 1.86 |
| 3 | 8 | 32256 | 0.0058 | 0.0087 (b-f) | 0.0082 (b-f) | 0.0123 | 0.0084 (b-f) | rot. | 1.50 |
| Σ | | | 0.0079 | 0.0430 | 0.0114 | 0.0162 | 0.0446 | best 0.0079 | |
| evaluate | | | 0.014 | 0.049 | 0.018 | 0.023 | 0.051 | | |

##### W1 (N = 7162), kD = 30, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0044 | 0.0036 (rot.) | 0.0040 (pw i2) | 0.0042 | 0.0073 (pw i2) | pw i2 | 1.00 |
| 3 | 15 | 32256 | 0.0241 | 0.0880 (b-f) | 0.0172 (pw i2) | 0.0179 | 0.0992 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.0285 | 0.0916 | 0.0212 | 0.0221 | 0.1064 | best 0.0221 | |
| evaluate | | | 0.036 | 0.100 | 0.030 | 0.031 | 0.121 | | |

##### W1 (N = 79578), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0182 | 0.0203 (rot.) | 0.0171 (pw i2) | 0.0183 | 0.0182 (pw i2) | rot. | 1.01 |
| 3 | 18 | 53352 | 0.0621 | 0.0630 (rot.) | 0.0409 (pw i2) | 0.0399 | 0.0403 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 385344 | 0.1177 | 0.0800 (b-f) | 0.0811 (b-f) | 0.1046 | 0.0821 (b-f) | b-f | 1.00 |
| 5 | 7 | 244944 | 0.0256 | 0.0087 (b-f) | 0.0087 (b-f) | 0.0697 | 0.0076 (b-f) | b-f | 1.00 |
| Σ | | | 0.2236 | 0.1719 | 0.1478 | 0.2325 | 0.1481 | best 0.1468 | |
| evaluate | | | 0.310 | 0.258 | 0.229 | 0.321 | 0.236 | | |

##### W1 (N = 79578), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0286 | 0.0316 (rot.) | 0.0245 (pw i2) | 0.0199 | 0.0174 (pw i2) | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.1089 | 0.1089 (rot.) | 0.0494 (pw i2) | 0.0493 | 0.0494 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.2551 | 0.2813 (b-f) | 0.1370 (pw i2) | 0.1331 | 0.2844 (b-f) | pw i2 | 1.00 |
| 5 | 15 | 244944 | 0.1757 | 0.2020 (b-f) | 0.1331 (pw i2) | 0.1330 | 0.2050 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.5684 | 0.6237 | 0.3439 | 0.3353 | 0.5562 | best 0.3353 | |
| evaluate | | | 0.767 | 0.825 | 0.548 | 0.539 | 0.760 | | |

##### W3 (N = 108849), kD = 30, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 12 | 3096 | 0.0016 | 0.0271 (b-f) | 0.0036 (pw i2) | 0.0035 | 0.0364 (b-f) | rot. | 2.16 |
| 3 | 8 | 53352 | 0.0083 | 0.0102 (b-f) | 0.0119 (b-f) | 0.0164 | 0.0105 (b-f) | rot. | 1.23 |
| 4 | 7 | 584136 | 0.0608 | 0.0225 (b-f) | 0.0196 (b-f) | 0.0727 | 0.0204 (b-f) | b-f | 1.00 |
| Σ | | | 0.0706 | 0.0598 | 0.0351 | 0.0927 | 0.0673 | best 0.0324 | |
| evaluate | | | 0.109 | 0.098 | 0.075 | 0.132 | 0.105 | | |

##### W3 (N = 108849), kD = 30, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 16 | 3096 | 0.0052 | 0.0049 (rot.) | 0.0047 (pw i2) | 0.0046 | 0.0049 (pw i2) | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0393 | 0.1107 (b-f) | 0.0219 (pw i2) | 0.0215 | 0.1068 (b-f) | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.3887 | 0.3958 (b-f) | 0.1230 (pw i2) | 0.1256 | 0.3917 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.4333 | 0.5114 | 0.1496 | 0.1517 | 0.5033 | best 0.1517 | |
| evaluate | | | 0.502 | 0.581 | 0.215 | 0.215 | 0.567 | | |

##### W3 (N = 4032), kD = 10, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0006 | 0.0050 (b-f) | 0.0050 (b-f) | 0.0018 | 0.0050 (b-f) | rot. | 8.55 |
| 3 | 7 | 5544 | 0.0008 | 0.0027 (b-f) | 0.0041 (b-f) | 0.0033 | 0.0037 (b-f) | rot. | 3.18 |
| Σ | | | 0.0014 | 0.0077 | 0.0091 | 0.0051 | 0.0087 | best 0.0014 | |
| evaluate | | | 0.015 | 0.021 | 0.022 | 0.019 | 0.022 | | |

##### W3 (N = 4032), kD = 10, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0037 | 0.1171 (b-f) | 0.0030 (pw i2) | 0.0024 | 0.1000 (b-f) | pw i2 | 1.00 |
| 3 | 15 | 5544 | 0.0051 | 0.0855 (b-f) | 0.0061 (pw i2) | 0.0048 | 0.0900 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.0088 | 0.2026 | 0.0091 | 0.0071 | 0.1900 | best 0.0071 | |
| evaluate | | | 0.042 | 0.235 | 0.044 | 0.040 | 0.223 | | |

##### W4 clusters (N = 100000), kD = 1, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0005 | 0.0047 (b-f) | 0.0051 (b-f) | 0.0014 | 0.0044 (b-f) | rot. | 9.03 |
| 3 | 7 | 19782 | 0.0024 | 0.0052 (b-f) | 0.0049 (b-f) | 0.0048 | 0.0052 (b-f) | rot. | 2.15 |
| 4 | 7 | 43392 | 0.0050 | 0.0057 (b-f) | 0.0060 (b-f) | 0.0094 | 0.0056 (b-f) | rot. | 1.14 |
| 5 | 7 | 106500 | 0.0116 | 0.0071 (b-f) | 0.0069 (b-f) | 0.0212 | 0.0069 (b-f) | b-f | 1.00 |
| 6 | 7 | 214818 | 0.0227 | 0.0100 (b-f) | 0.0097 (b-f) | 0.0356 | 0.0093 (b-f) | b-f | 1.00 |
| 7 | 7 | 289194 | 0.0320 | 0.0146 (b-f) | 0.0130 (b-f) | 0.0385 | 0.0134 (b-f) | b-f | 1.00 |
| 8 | 7 | 351936 | 0.0373 | 0.0158 (b-f) | 0.0176 (b-f) | 0.0475 | 0.0165 (b-f) | b-f | 1.00 |
| Σ | | | 0.1115 | 0.0631 | 0.0632 | 0.1584 | 0.0615 | best 0.0555 | |
| evaluate | | | 0.268 | 0.218 | 0.217 | 0.315 | 0.213 | | |

##### W4 clusters (N = 100000), kD = 1, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0033 | 0.0033 (rot.) | 0.0022 (pw i2) | 0.0035 | 0.0029 (rot.) | rot. | 1.06 |
| 3 | 15 | 19782 | 0.0152 | 0.0153 (rot.) | 0.0089 (pw i2) | 0.0107 | 0.0150 (rot.) | pw i2 | 1.00 |
| 4 | 15 | 43392 | 0.0302 | 0.0312 (rot.) | 0.0146 (pw i2) | 0.0166 | 0.0306 (rot.) | pw i2 | 1.00 |
| 5 | 15 | 106500 | 0.0750 | 0.0730 (rot.) | 0.0319 (pw i2) | 0.0331 | 0.0728 (rot.) | pw i2 | 1.00 |
| 6 | 15 | 214818 | 0.1470 | 0.1873 (b-f) | 0.0582 (pw i2) | 0.0589 | 0.1895 (b-f) | pw i2 | 1.00 |
| 7 | 15 | 289194 | 0.1942 | 0.2824 (b-f) | 0.0700 (pw i2) | 0.0709 | 0.2853 (b-f) | pw i2 | 1.00 |
| 8 | 15 | 351936 | 0.2374 | 0.3190 (b-f) | 0.0820 (pw i2) | 0.0820 | 0.2787 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.7024 | 0.9115 | 0.2679 | 0.2756 | 0.8748 | best 0.2754 | |
| evaluate | | | 1.123 | 1.325 | 0.678 | 0.688 | 1.297 | | |

##### W4 clusters (N = 100000), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 21 | 3096 | 0.0071 | 0.0074 (rot.) | 0.0084 (pw i2) | 0.0082 | 0.0089 (pw i2) | rot. | 1.15 |
| 3 | 13 | 19782 | 0.0103 | 0.0593 (b-f) | 0.0136 (pw i2) | 0.0137 | 0.0582 (b-f) | rot. | 1.33 |
| 4 | 8 | 43392 | 0.0066 | 0.0089 (b-f) | 0.0101 (b-f) | 0.0159 | 0.0105 (b-f) | rot. | 1.36 |
| 5 | 7 | 106500 | 0.0115 | 0.0068 (b-f) | 0.0072 (b-f) | 0.0244 | 0.0068 (b-f) | b-f | 1.00 |
| 6 | 7 | 214818 | 0.0227 | 0.0093 (b-f) | 0.0098 (b-f) | 0.0369 | 0.0084 (b-f) | b-f | 1.00 |
| 7 | 7 | 289194 | 0.0299 | 0.0145 (b-f) | 0.0129 (b-f) | 0.0464 | 0.0133 (b-f) | b-f | 1.00 |
| 8 | 7 | 351936 | 0.0364 | 0.0146 (b-f) | 0.0176 (b-f) | 0.0509 | 0.0160 (b-f) | b-f | 1.00 |
| Σ | | | 0.1245 | 0.1208 | 0.0795 | 0.1964 | 0.1221 | best 0.0691 | |
| evaluate | | | 0.281 | 0.277 | 0.242 | 0.359 | 0.281 | | |

##### W4 clusters (N = 100000), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 25 | 3096 | 0.0127 | 0.0113 (rot.) | 0.0086 (pw i2) | 0.0090 | 0.0087 (pw i2) | pw i2 | 1.00 |
| 3 | 16 | 19782 | 0.0177 | 0.0181 (rot.) | 0.0158 (pw i2) | 0.0146 | 0.0154 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 43392 | 0.0326 | 0.0305 (rot.) | 0.0219 (pw i2) | 0.0208 | 0.0322 (rot.) | pw i2 | 1.00 |
| 5 | 15 | 106500 | 0.0727 | 0.0749 (rot.) | 0.0383 (pw i2) | 0.0382 | 0.0739 (rot.) | pw i2 | 1.00 |
| 6 | 15 | 214818 | 0.1436 | 0.1960 (b-f) | 0.0605 (pw i2) | 0.0629 | 0.1773 (b-f) | pw i2 | 1.00 |
| 7 | 15 | 289194 | 0.1942 | 0.2756 (b-f) | 0.0740 (pw i2) | 0.0724 | 0.2734 (b-f) | pw i2 | 1.00 |
| 8 | 15 | 351936 | 0.2372 | 0.2937 (b-f) | 0.0850 (pw i2) | 0.0855 | 0.2986 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.7107 | 0.9000 | 0.3042 | 0.3034 | 0.8795 | best 0.3034 | |
| evaluate | | | 1.125 | 1.315 | 0.719 | 0.717 | 1.287 | | |

##### W4 cube (N = 100000), kD = 1, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0005 | 0.0045 (b-f) | 0.0044 (b-f) | 0.0017 | 0.0047 (b-f) | rot. | 8.61 |
| 3 | 7 | 53352 | 0.0060 | 0.0064 (b-f) | 0.0067 (b-f) | 0.0118 | 0.0063 (b-f) | rot. | 1.07 |
| 4 | 7 | 584136 | 0.0605 | 0.0219 (b-f) | 0.0225 (b-f) | 0.0663 | 0.0204 (b-f) | b-f | 1.00 |
| Σ | | | 0.0670 | 0.0328 | 0.0336 | 0.0798 | 0.0314 | best 0.0285 | |
| evaluate | | | 0.101 | 0.069 | 0.069 | 0.116 | 0.067 | | |

##### W4 cube (N = 100000), kD = 1, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0036 | 0.1077 (b-f) | 0.0029 (pw i2) | 0.0030 | 0.0997 (b-f) | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0393 | 0.1110 (b-f) | 0.0163 (pw i2) | 0.0162 | 0.1137 (b-f) | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.3847 | 0.3791 (b-f) | 0.1165 (pw i2) | 0.1148 | 0.4031 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.4276 | 0.5978 | 0.1357 | 0.1341 | 0.6164 | best 0.1341 | |
| evaluate | | | 0.489 | 0.657 | 0.196 | 0.197 | 0.679 | | |

##### W4 cube (N = 100000), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0192 | 0.0196 (rot.) | 0.0197 (pw i2) | 0.0184 | 0.0177 (pw i2) | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.0618 | 0.0607 (rot.) | 0.0420 (pw i2) | 0.0427 | 0.0417 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 584136 | 0.1741 | 0.1069 (b-f) | 0.1113 (b-f) | 0.1349 | 0.1116 (b-f) | b-f | 1.00 |
| Σ | | | 0.2551 | 0.1872 | 0.1729 | 0.1960 | 0.1710 | best 0.1680 | |
| evaluate | | | 0.311 | 0.242 | 0.228 | 0.250 | 0.226 | | |

##### W4 cube (N = 100000), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0319 | 0.0303 (rot.) | 0.0189 (pw i2) | 0.0179 | 0.0183 (pw i2) | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.1095 | 0.1093 (rot.) | 0.0509 (pw i2) | 0.0498 | 0.0506 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.3852 | 0.3733 (b-f) | 0.1688 (pw i2) | 0.1752 | 0.3765 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.5266 | 0.5128 | 0.2386 | 0.2430 | 0.4454 | best 0.2430 | |
| evaluate | | | 0.605 | 0.595 | 0.317 | 0.322 | 0.525 | | |

##### W4 plummer (N = 100000), kD = 1, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0005 | 0.0049 (b-f) | 0.0053 (b-f) | 0.0017 | 0.0049 (b-f) | rot. | 9.24 |
| 3 | 7 | 53352 | 0.0061 | 0.0064 (b-f) | 0.0065 (b-f) | 0.0103 | 0.0062 (b-f) | rot. | 1.04 |
| 4 | 7 | 53352 | 0.0062 | 0.0056 (b-f) | 0.0058 (b-f) | 0.0110 | 0.0067 (b-f) | b-f | 1.00 |
| 5 | 7 | 109548 | 0.0118 | 0.0086 (b-f) | 0.0079 (b-f) | 0.0202 | 0.0081 (b-f) | b-f | 1.00 |
| 6 | 7 | 205458 | 0.0218 | 0.0103 (b-f) | 0.0097 (b-f) | 0.0283 | 0.0103 (b-f) | b-f | 1.00 |
| 7 | 7 | 333330 | 0.0347 | 0.0133 (b-f) | 0.0130 (b-f) | 0.0415 | 0.0115 (b-f) | b-f | 1.00 |
| 8 | 7 | 35148 | 0.0041 | 0.0055 (b-f) | 0.0054 (b-f) | 0.0084 | 0.0061 (b-f) | rot. | 1.33 |
| Σ | | | 0.0852 | 0.0546 | 0.0536 | 0.1213 | 0.0538 | best 0.0486 | |
| evaluate | | | 0.240 | 0.216 | 0.208 | 0.282 | 0.210 | | |

##### W4 plummer (N = 100000), kD = 1, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0033 | 0.0036 (rot.) | 0.0030 (pw i2) | 0.0028 | 0.0036 (rot.) | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0387 | 0.0392 (rot.) | 0.0169 (pw i2) | 0.0177 | 0.0392 (rot.) | pw i2 | 1.00 |
| 4 | 15 | 53352 | 0.0398 | 0.0380 (rot.) | 0.0162 (pw i2) | 0.0169 | 0.0376 (rot.) | pw i2 | 1.00 |
| 5 | 15 | 109548 | 0.0765 | 0.0767 (rot.) | 0.0285 (pw i2) | 0.0279 | 0.0742 (rot.) | pw i2 | 1.00 |
| 6 | 15 | 205458 | 0.1378 | 0.1748 (b-f) | 0.0462 (pw i2) | 0.0483 | 0.1993 (b-f) | pw i2 | 1.00 |
| 7 | 15 | 333330 | 0.2258 | 0.2093 (b-f) | 0.0718 (pw i2) | 0.0722 | 0.2060 (b-f) | pw i2 | 1.00 |
| 8 | 15 | 35148 | 0.0253 | 0.1307 (b-f) | 0.0136 (pw i2) | 0.0134 | 0.1191 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.5472 | 0.6723 | 0.1962 | 0.1991 | 0.6789 | best 0.1991 | |
| evaluate | | | 0.960 | 1.082 | 0.612 | 0.622 | 1.098 | | |

##### W4 plummer (N = 100000), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0212 | 0.0208 (rot.) | 0.0170 (pw i2) | 0.0187 | 0.0162 (pw i2) | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.0610 | 0.0636 (rot.) | 0.0425 (pw i2) | 0.0416 | 0.0411 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 380178 | 0.1154 | 0.0859 (b-f) | 0.0886 (b-f) | 0.0922 | 0.0912 (b-f) | b-f | 1.00 |
| 5 | 7 | 109548 | 0.0119 | 0.0078 (b-f) | 0.0076 (b-f) | 0.0256 | 0.0069 (b-f) | b-f | 1.00 |
| 6 | 7 | 205458 | 0.0219 | 0.0099 (b-f) | 0.0103 (b-f) | 0.0313 | 0.0092 (b-f) | b-f | 1.00 |
| 7 | 7 | 333330 | 0.0348 | 0.0103 (b-f) | 0.0103 (b-f) | 0.0443 | 0.0127 (b-f) | b-f | 1.00 |
| 8 | 7 | 35148 | 0.0049 | 0.0066 (b-f) | 0.0062 (b-f) | 0.0081 | 0.0058 (b-f) | rot. | 1.33 |
| Σ | | | 0.2712 | 0.2049 | 0.1825 | 0.2618 | 0.1832 | best 0.1791 | |
| evaluate | | | 0.438 | 0.369 | 0.349 | 0.430 | 0.349 | | |

##### W4 plummer (N = 100000), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0299 | 0.0299 (rot.) | 0.0217 (pw i2) | 0.0201 | 0.0172 (pw i2) | pw i2 | 1.00 |
| 3 | 21 | 53352 | 0.0931 | 0.0950 (rot.) | 0.0457 (pw i2) | 0.0491 | 0.0467 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 380178 | 0.2517 | 0.2517 (rot.) | 0.1131 (pw i2) | 0.1165 | 0.2519 (rot.) | pw i2 | 1.00 |
| 5 | 15 | 109548 | 0.0748 | 0.0779 (rot.) | 0.0349 (pw i2) | 0.0370 | 0.0754 (rot.) | pw i2 | 1.00 |
| 6 | 15 | 205458 | 0.1390 | 0.1743 (b-f) | 0.0504 (pw i2) | 0.0519 | 0.1789 (b-f) | pw i2 | 1.00 |
| 7 | 15 | 333330 | 0.2243 | 0.2132 (b-f) | 0.0767 (pw i2) | 0.0782 | 0.2230 (b-f) | pw i2 | 1.00 |
| 8 | 15 | 35148 | 0.0249 | 0.1356 (b-f) | 0.0147 (pw i2) | 0.0141 | 0.1417 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.8376 | 0.9776 | 0.3572 | 0.3669 | 0.9347 | best 0.3669 | |
| evaluate | | | 1.251 | 1.385 | 0.761 | 0.779 | 1.338 | | |

##### W4 sphere (N = 100000), kD = 1, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 7 | 3096 | 0.0006 | 0.0047 (b-f) | 0.0047 (b-f) | 0.0014 | 0.0048 (b-f) | rot. | 8.54 |
| 3 | 7 | 53352 | 0.0061 | 0.0062 (b-f) | 0.0064 (b-f) | 0.0103 | 0.0063 (b-f) | rot. | 1.03 |
| 4 | 7 | 385344 | 0.0397 | 0.0146 (b-f) | 0.0137 (b-f) | 0.0539 | 0.0146 (b-f) | b-f | 1.00 |
| 5 | 7 | 325404 | 0.0371 | 0.0153 (b-f) | 0.0140 (b-f) | 0.0667 | 0.0135 (b-f) | b-f | 1.00 |
| Σ | | | 0.0834 | 0.0409 | 0.0388 | 0.1323 | 0.0393 | best 0.0366 | |
| evaluate | | | 0.159 | 0.116 | 0.112 | 0.209 | 0.113 | | |

##### W4 sphere (N = 100000), kD = 1, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 15 | 3096 | 0.0033 | 0.0044 (rot.) | 0.0028 (pw i2) | 0.0027 | 0.0040 (rot.) | pw i2 | 1.00 |
| 3 | 15 | 53352 | 0.0379 | 0.1123 (b-f) | 0.0161 (pw i2) | 0.0154 | 0.1183 (b-f) | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.2543 | 0.2975 (b-f) | 0.0861 (pw i2) | 0.0848 | 0.2905 (b-f) | pw i2 | 1.00 |
| 5 | 15 | 325404 | 0.2146 | 0.2722 (b-f) | 0.1197 (pw i2) | 0.1201 | 0.2715 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.5101 | 0.6864 | 0.2247 | 0.2230 | 0.6844 | best 0.2230 | |
| evaluate | | | 0.697 | 0.872 | 0.407 | 0.407 | 0.865 | | |

##### W4 sphere (N = 100000), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0186 | 0.0224 (rot.) | 0.0178 (pw i2) | 0.0180 | 0.0195 (pw i2) | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.0620 | 0.0622 (rot.) | 0.0406 (pw i2) | 0.0409 | 0.0410 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 385344 | 0.1148 | 0.0790 (b-f) | 0.0807 (b-f) | 0.1039 | 0.0821 (b-f) | b-f | 1.00 |
| 5 | 7 | 325404 | 0.0342 | 0.0127 (b-f) | 0.0137 (b-f) | 0.0823 | 0.0140 (b-f) | b-f | 1.00 |
| Σ | | | 0.2297 | 0.1763 | 0.1528 | 0.2451 | 0.1566 | best 0.1506 | |
| evaluate | | | 0.321 | 0.267 | 0.237 | 0.336 | 0.248 | | |

##### W4 sphere (N = 100000), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0304 | 0.0316 (rot.) | 0.0188 (pw i2) | 0.0196 | 0.0200 (pw i2) | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.1091 | 0.1093 (rot.) | 0.0486 (pw i2) | 0.0511 | 0.0475 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 385344 | 0.2722 | 0.2800 (b-f) | 0.1291 (pw i2) | 0.1293 | 0.2992 (b-f) | pw i2 | 1.00 |
| 5 | 15 | 325404 | 0.2159 | 0.2727 (b-f) | 0.1593 (pw i2) | 0.1529 | 0.2789 (b-f) | pw i2 | 1.00 |
| Σ | | | 0.6277 | 0.6936 | 0.3558 | 0.3529 | 0.6456 | best 0.3529 | |
| evaluate | | | 0.829 | 0.888 | 0.551 | 0.547 | 0.839 | | |

#### kathleen-large

##### W1 (N = 1989437), kD = 500, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 123 | 3096 | 2.6521 | 2.7822 (rot.) | 2.8036 (rot.) | 2.7586 (rot.) | 2.7344 (rot.) | rot. | 1.00 |
| 3 | 66 | 53352 | 3.4263 | 3.4121 (rot.) | 3.5694 (rot.) | 3.5242 (rot.) | 3.4430 (rot.) | rot. | 1.00 |
| 4 | 36 | 511200 | 3.9927 | 4.0592 (rot.) | 1.3266 (pw i2) | 1.3284 | 1.3030 (pw i2) | pw i2 | 1.00 |
| 5 | 21 | 2935296 | 4.8755 | 4.8711 (rot.) | 1.7392 (pw i2) | 1.7406 | 1.7348 (pw i2) | pw i2 | 1.00 |
| 6 | 13 | 10439280 | 4.7133 | 3.6068 (b-f) | 2.3854 (pw i2) | 2.3906 | 3.6068 (b-f) | pw i2 | 1.00 |
| 7 | 8 | 14492220 | 2.0144 | 0.8840 (b-f) | 0.8768 (b-f) | 2.5796 | 0.8913 (b-f) | b-f | 1.00 |
| Σ | | | 21.6744 | 19.6154 | 12.7009 | 14.3220 | 13.7133 | best 12.4220 | |
| evaluate | | | 23.348 | 21.282 | 14.347 | 16.091 | 15.369 | | |

##### W1 (N = 1989437), kD = 500, ε = 1e-3, 4 × 10

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 123 | 3096 | 4.3301 | 4.2438 (rot.) | 4.4560 (rot.) | 4.2275 (rot.) | 4.7219 (rot.) | rot. | 1.00 |
| 3 | 66 | 53352 | 3.4221 | 3.3571 (rot.) | 3.5227 (rot.) | 3.4999 (rot.) | 3.3019 (rot.) | rot. | 1.00 |
| 4 | 36 | 511200 | 4.1131 | 4.1187 (rot.) | 1.5837 (pw i2) | 1.5507 | 1.6456 (pw i2) | pw i2 | 1.00 |
| 5 | 21 | 2935296 | 4.9460 | 6.9963 (rot.) | 1.8610 (pw i2) | 1.8962 | 2.0107 (pw i2) | pw i2 | 1.00 |
| 6 | 13 | 10439280 | 4.7447 | 3.4867 (b-f) | 2.4428 (pw i2) | 2.4585 | 3.3667 (b-f) | pw i2 | 1.00 |
| 7 | 8 | 14492220 | 2.0144 | 0.8333 (b-f) | 0.7971 (b-f) | 2.5941 | 0.8116 (b-f) | b-f | 1.00 |
| Σ | | | 23.5703 | 23.0359 | 14.6632 | 16.2269 | 15.8583 | best 14.4908 | |
| evaluate | | | 24.973 | 24.197 | 15.688 | 17.463 | 16.829 | | |

##### W1 (N = 1989437), kD = 500, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 3.2185 | 3.1093 (rot.) | 3.2050 (rot.) | 3.2010 (rot.) | 3.2311 (rot.) | rot. | 1.00 |
| 3 | 72 | 53352 | 4.6664 | 4.8150 (rot.) | 4.7327 (rot.) | 4.6355 (rot.) | 4.8843 (rot.) | rot. | 1.00 |
| 4 | 41 | 511200 | 5.9729 | 5.8962 (rot.) | 1.4682 (pw i2) | 1.4636 | 1.4567 (pw i2) | pw i2 | 1.00 |
| 5 | 25 | 2935296 | 8.0735 | 8.0926 (rot.) | 2.0077 (pw i2) | 2.0474 | 2.0033 (pw i2) | pw i2 | 1.00 |
| 6 | 16 | 10439280 | 8.1322 | 8.1635 (rot.) | 2.8760 (pw i2) | 2.9021 | 2.8760 (pw i2) | pw i2 | 1.00 |
| 7 | 15 | 14492220 | 9.5359 | 11.1373 (b-f) | 4.2680 (pw i2) | 4.2825 | 12.2604 (b-f) | pw i2 | 1.00 |
| Σ | | | 39.5993 | 41.2138 | 18.5577 | 18.5320 | 26.7119 | best 18.5804 | |
| evaluate | | | 42.397 | 44.052 | 21.310 | 21.225 | 29.369 | | |

##### W1 (N = 1989437), kD = 500, ε = 1e-6, 4 × 10

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 131 | 3096 | 5.3623 | 5.1292 (rot.) | 5.1536 (rot.) | 5.1175 (rot.) | 5.4421 (rot.) | rot. | 1.00 |
| 3 | 72 | 53352 | 4.5591 | 4.6847 (rot.) | 4.4628 (rot.) | 4.6960 (rot.) | 4.5604 (rot.) | rot. | 1.00 |
| 4 | 41 | 511200 | 6.1860 | 6.1497 (rot.) | 1.8104 (pw i2) | 1.6969 | 1.8904 (pw i2) | pw i2 | 1.00 |
| 5 | 25 | 2935296 | 9.6747 | 8.0970 (rot.) | 2.1501 (pw i2) | 2.1560 | 2.1428 (pw i2) | pw i2 | 1.00 |
| 6 | 16 | 10439280 | 8.1583 | 8.2523 (rot.) | 2.9387 (pw i2) | 2.9700 | 3.1057 (pw i2) | pw i2 | 1.00 |
| 7 | 15 | 14492220 | 10.2677 | 10.0648 (b-f) | 4.3767 (pw i2) | 4.4346 | 9.8547 (b-f) | pw i2 | 1.00 |
| Σ | | | 44.2083 | 42.3778 | 20.8922 | 21.0710 | 26.9960 | best 21.1790 | |
| evaluate | | | 45.038 | 44.325 | 23.013 | 23.076 | 28.559 | | |

##### W3 (N = 4031442), kD = 100, ε = 1e-3, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0195 | 0.0187 (rot.) | 0.0177 (pw i2) | 0.0233 | 0.0202 (pw i2) | rot. | 1.19 |
| 3 | 18 | 53352 | 0.0616 | 0.0613 (rot.) | 0.0394 (pw i2) | 0.0414 | 0.0418 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 584136 | 0.1714 | 0.1063 (b-f) | 0.1066 (b-f) | 0.1259 | 0.1066 (b-f) | b-f | 1.00 |
| 5 | 7 | 5398920 | 0.5399 | 0.1431 (b-f) | 0.1458 (b-f) | 0.4832 | 0.1458 (b-f) | b-f | 1.00 |
| 6 | 7 | 46298376 | 4.7224 | 1.2269 (b-f) | 1.2269 (b-f) | 3.5881 | 1.2269 (b-f) | b-f | 1.00 |
| Σ | | | 5.5149 | 1.5563 | 1.5364 | 4.2619 | 1.5413 | best 1.5372 | |
| evaluate | | | 6.880 | 2.873 | 2.855 | 5.603 | 2.849 | | |

##### W3 (N = 4031442), kD = 100, ε = 1e-3, 4 × 10

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 30 | 3096 | 0.0343 | 0.0339 (rot.) | 0.0338 (pw i2) | 0.0342 | 0.0324 (pw i2) | pw i2 | 1.00 |
| 3 | 18 | 53352 | 0.0611 | 0.0615 (rot.) | 0.0505 (pw i2) | 0.0490 | 0.0525 (pw i2) | pw i2 | 1.00 |
| 4 | 11 | 584136 | 0.1738 | 0.1057 (b-f) | 0.1101 (b-f) | 0.1288 | 0.1101 (b-f) | b-f | 1.00 |
| 5 | 7 | 5398920 | 0.5534 | 0.1593 (b-f) | 0.1620 (b-f) | 0.4967 | 0.1647 (b-f) | b-f | 1.00 |
| 6 | 7 | 46298376 | 5.4864 | 1.5741 (b-f) | 1.4121 (b-f) | 3.6344 | 1.6204 (b-f) | b-f | 1.00 |
| Σ | | | 6.3089 | 1.9346 | 1.7684 | 4.3431 | 1.9801 | best 1.9224 | |
| evaluate | | | 7.204 | 2.831 | 2.689 | 5.283 | 2.841 | | |

##### W3 (N = 4031442), kD = 100, ε = 1e-6, 1 × 40

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0287 | 0.0293 (rot.) | 0.0240 (pw i2) | 0.0238 | 0.0247 (pw i2) | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.1056 | 0.1069 (rot.) | 0.0503 (pw i2) | 0.0482 | 0.0468 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.3858 | 0.3695 (b-f) | 0.1674 (pw i2) | 0.1671 | 0.3826 (b-f) | pw i2 | 1.00 |
| 5 | 15 | 5398920 | 3.7334 | 3.0396 (b-f) | 0.8368 (pw i2) | 0.8395 | 3.0450 (b-f) | pw i2 | 1.00 |
| 6 | 15 | 46298376 | 32.7793 | 24.8854 (b-f) | 6.2734 (pw i2) | 6.2040 | 24.6076 (b-f) | pw i2 | 1.00 |
| Σ | | | 37.0327 | 28.4306 | 7.3519 | 7.2826 | 28.1066 | best 7.2826 | |
| evaluate | | | 39.752 | 31.242 | 9.825 | 9.818 | 30.798 | | |

##### W3 (N = 4031442), kD = 100, ε = 1e-6, 4 × 10

| level | p | V pairs | rotation | auto | auto-pw | planewave | auto-pw:16 | fastest | auto-pw ÷ fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 35 | 3096 | 0.0533 | 0.0523 (rot.) | 0.0365 (pw i2) | 0.0378 | 0.0369 (pw i2) | pw i2 | 1.00 |
| 3 | 22 | 53352 | 0.1069 | 0.1099 (rot.) | 0.0572 (pw i2) | 0.0579 | 0.0584 (pw i2) | pw i2 | 1.00 |
| 4 | 15 | 584136 | 0.3849 | 0.4080 (b-f) | 0.1668 (pw i2) | 0.1697 | 0.3581 (b-f) | pw i2 | 1.00 |
| 5 | 15 | 5398920 | 3.7118 | 3.1098 (b-f) | 0.9070 (pw i2) | 0.9178 | 3.1206 (b-f) | pw i2 | 1.00 |
| 6 | 15 | 46298376 | 34.7469 | 28.9596 (b-f) | 6.4818 (pw i2) | 6.4818 | 26.0660 (b-f) | pw i2 | 1.00 |
| Σ | | | 39.0038 | 32.6396 | 7.6492 | 7.6650 | 29.6400 | best 7.6650 | |
| evaluate | | | 40.812 | 34.589 | 9.740 | 9.659 | 31.421 | | |

#### Summary

| machine | workload | kD | ε | split | M2L Σ rotation | auto | auto-pw | planewave | auto-pw:16 | best per level | auto-pw/best | auto-pw/auto | evaluate rotation | auto | auto-pw | planewave | auto-pw:16 | auto-pw/auto | ABBA max | auto-pw ÷ rot. error φ/∇φ | auto-pw's checksum also |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| kathleen | W1 | 10 | 1e-3 | 1×40 | 0.001 | 0.005 | 0.005 | 0.002 | 0.005 | 0.001 | 8.82 | 0.95 | 0.002 | 0.007 | 0.006 | 0.004 | 0.007 | 0.91 | 1.51 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W1 | 10 | 1e-6 | 1×40 | 0.003 | 0.114 | 0.003 | 0.002 | 0.104 | 0.002 | 1.51 | 0.03 | 0.005 | 0.118 | 0.006 | 0.004 | 0.106 | 0.05 | 1.35 | 1.005/0.995 | planewave |
| kathleen | W1 | 30 | 1e-3 | 1×40 | 0.008 | 0.043 | 0.011 | 0.016 | 0.045 | 0.008 | 1.44 | 0.26 | 0.014 | 0.049 | 0.018 | 0.023 | 0.051 | 0.35 | 1.50 | 1.000/1.000 | – |
| kathleen | W1 | 30 | 1e-6 | 1×40 | 0.028 | 0.092 | 0.021 | 0.022 | 0.106 | 0.022 | 0.96 | 0.23 | 0.036 | 0.100 | 0.030 | 0.031 | 0.121 | 0.30 | 2.39 | 1.000/1.000 | planewave |
| kathleen | W1 | 100 | 1e-3 | 1×40 | 0.224 | 0.172 | 0.148 | 0.233 | 0.148 | 0.147 | 1.01 | 0.86 | 0.310 | 0.258 | 0.229 | 0.321 | 0.236 | 0.89 | 1.30 | 1.000/1.000 | auto-pw:16 |
| kathleen | W1 | 100 | 1e-6 | 1×40 | 0.568 | 0.624 | 0.344 | 0.335 | 0.556 | 0.335 | 1.03 | 0.55 | 0.767 | 0.825 | 0.548 | 0.539 | 0.760 | 0.66 | 1.64 | 0.995/0.980 | planewave |
| kathleen | W3 (N = 108849) | 30 | 1e-3 | 1×40 | 0.071 | 0.060 | 0.035 | 0.093 | 0.067 | 0.032 | 1.08 | 0.59 | 0.109 | 0.098 | 0.075 | 0.132 | 0.105 | 0.76 | 1.55 | 1.000/1.000 | – |
| kathleen | W3 (N = 108849) | 30 | 1e-6 | 1×40 | 0.433 | 0.511 | 0.150 | 0.152 | 0.503 | 0.152 | 0.99 | 0.29 | 0.502 | 0.581 | 0.215 | 0.215 | 0.567 | 0.37 | 1.07 | 0.973/0.964 | planewave |
| kathleen | W3 (N = 4032) | 10 | 1e-3 | 1×40 | 0.001 | 0.008 | 0.009 | 0.005 | 0.009 | 0.001 | 6.39 | 1.19 | 0.015 | 0.021 | 0.022 | 0.019 | 0.022 | 1.04 | 1.33 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W3 (N = 4032) | 10 | 1e-6 | 1×40 | 0.009 | 0.203 | 0.009 | 0.007 | 0.190 | 0.007 | 1.27 | 0.04 | 0.042 | 0.235 | 0.044 | 0.040 | 0.223 | 0.19 | 1.45 | 0.991/0.947 | planewave |
| kathleen | W4 clusters (N = 100000) | 1 | 1e-3 | 1×40 | 0.111 | 0.063 | 0.063 | 0.158 | 0.062 | 0.055 | 1.14 | 1.00 | 0.268 | 0.218 | 0.217 | 0.315 | 0.213 | 1.00 | 1.29 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W4 clusters (N = 100000) | 1 | 1e-6 | 1×40 | 0.702 | 0.911 | 0.268 | 0.276 | 0.875 | 0.275 | 0.97 | 0.29 | 1.123 | 1.325 | 0.678 | 0.688 | 1.297 | 0.51 | 1.28 | 0.969/0.916 | planewave |
| kathleen | W4 clusters (N = 100000) | 100 | 1e-3 | 1×40 | 0.124 | 0.121 | 0.080 | 0.196 | 0.122 | 0.069 | 1.15 | 0.66 | 0.281 | 0.277 | 0.242 | 0.359 | 0.281 | 0.88 | 1.35 | 1.000/1.000 | – |
| kathleen | W4 clusters (N = 100000) | 100 | 1e-6 | 1×40 | 0.711 | 0.900 | 0.304 | 0.303 | 0.880 | 0.303 | 1.00 | 0.34 | 1.125 | 1.315 | 0.719 | 0.717 | 1.287 | 0.55 | 1.15 | 0.970/0.921 | planewave |
| kathleen | W4 cube (N = 100000) | 1 | 1e-3 | 1×40 | 0.067 | 0.033 | 0.034 | 0.080 | 0.031 | 0.028 | 1.18 | 1.02 | 0.101 | 0.069 | 0.069 | 0.116 | 0.067 | 1.00 | 1.21 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W4 cube (N = 100000) | 1 | 1e-6 | 1×40 | 0.428 | 0.598 | 0.136 | 0.134 | 0.616 | 0.134 | 1.01 | 0.23 | 0.489 | 0.657 | 0.196 | 0.197 | 0.679 | 0.30 | 1.42 | 0.983/0.928 | planewave |
| kathleen | W4 cube (N = 100000) | 100 | 1e-3 | 1×40 | 0.255 | 0.187 | 0.173 | 0.196 | 0.171 | 0.168 | 1.03 | 0.92 | 0.311 | 0.242 | 0.228 | 0.250 | 0.226 | 0.94 | 1.17 | 1.000/1.000 | auto-pw:16 |
| kathleen | W4 cube (N = 100000) | 100 | 1e-6 | 1×40 | 0.527 | 0.513 | 0.239 | 0.243 | 0.445 | 0.243 | 0.98 | 0.47 | 0.605 | 0.595 | 0.317 | 0.322 | 0.525 | 0.53 | 1.16 | 1.003/0.990 | planewave |
| kathleen | W4 plummer (N = 100000) | 1 | 1e-3 | 1×40 | 0.085 | 0.055 | 0.054 | 0.121 | 0.054 | 0.049 | 1.10 | 0.98 | 0.240 | 0.216 | 0.208 | 0.282 | 0.210 | 0.97 | 1.35 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W4 plummer (N = 100000) | 1 | 1e-6 | 1×40 | 0.547 | 0.672 | 0.196 | 0.199 | 0.679 | 0.199 | 0.99 | 0.29 | 0.960 | 1.082 | 0.612 | 0.622 | 1.098 | 0.57 | 1.34 | 1.063/0.940 | planewave |
| kathleen | W4 plummer (N = 100000) | 100 | 1e-3 | 1×40 | 0.271 | 0.205 | 0.183 | 0.262 | 0.183 | 0.179 | 1.02 | 0.89 | 0.438 | 0.369 | 0.349 | 0.430 | 0.349 | 0.95 | 1.38 | 1.000/1.000 | auto-pw:16 |
| kathleen | W4 plummer (N = 100000) | 100 | 1e-6 | 1×40 | 0.838 | 0.978 | 0.357 | 0.367 | 0.935 | 0.367 | 0.97 | 0.37 | 1.251 | 1.385 | 0.761 | 0.779 | 1.338 | 0.55 | 1.25 | 1.013/0.955 | planewave |
| kathleen | W4 sphere (N = 100000) | 1 | 1e-3 | 1×40 | 0.083 | 0.041 | 0.039 | 0.132 | 0.039 | 0.037 | 1.06 | 0.95 | 0.159 | 0.116 | 0.112 | 0.209 | 0.113 | 0.97 | 1.30 | 1.000/1.000 | auto,auto-pw:16 |
| kathleen | W4 sphere (N = 100000) | 1 | 1e-6 | 1×40 | 0.510 | 0.686 | 0.225 | 0.223 | 0.684 | 0.223 | 1.01 | 0.33 | 0.697 | 0.872 | 0.407 | 0.407 | 0.865 | 0.47 | 1.25 | 0.986/0.945 | planewave |
| kathleen | W4 sphere (N = 100000) | 100 | 1e-3 | 1×40 | 0.230 | 0.176 | 0.153 | 0.245 | 0.157 | 0.151 | 1.01 | 0.87 | 0.321 | 0.267 | 0.237 | 0.336 | 0.248 | 0.89 | 1.36 | 1.000/1.000 | auto-pw:16 |
| kathleen | W4 sphere (N = 100000) | 100 | 1e-6 | 1×40 | 0.628 | 0.694 | 0.356 | 0.353 | 0.646 | 0.353 | 1.01 | 0.51 | 0.829 | 0.888 | 0.551 | 0.547 | 0.839 | 0.62 | 1.15 | 1.003/0.978 | planewave |
| kathleen-large | W1 | 500 | 1e-3 | 1×40 | 21.674 | 19.615 | 12.701 | 14.322 | 13.713 | 12.422 | 1.02 | 0.65 | 23.348 | 21.282 | 14.347 | 16.091 | 15.369 | 0.67 | 1.22 | 1.000/1.000 | – |
| kathleen-large | W1 | 500 | 1e-3 | 4×10 | 23.570 | 23.036 | 14.663 | 16.227 | 15.858 | 14.491 | 1.01 | 0.64 | 24.973 | 24.197 | 15.688 | 17.463 | 16.829 | 0.65 | 1.17 | 1.000/1.000 | – |
| kathleen-large | W1 | 500 | 1e-6 | 1×40 | 39.599 | 41.214 | 18.558 | 18.532 | 26.712 | 18.580 | 1.00 | 0.45 | 42.397 | 44.052 | 21.310 | 21.225 | 29.369 | 0.48 | 1.17 | 0.973/0.984 | planewave |
| kathleen-large | W1 | 500 | 1e-6 | 4×10 | 44.208 | 42.378 | 20.892 | 21.071 | 26.996 | 21.179 | 0.99 | 0.49 | 45.038 | 44.325 | 23.013 | 23.076 | 28.559 | 0.52 | 1.40 | 0.973/0.984 | planewave |
| kathleen-large | W3 (N = 4031442) | 100 | 1e-3 | 1×40 | 5.515 | 1.556 | 1.536 | 4.262 | 1.541 | 1.537 | 1.00 | 0.99 | 6.880 | 2.873 | 2.855 | 5.603 | 2.849 | 0.99 | 1.80 | 1.000/1.000 | auto-pw:16 |
| kathleen-large | W3 (N = 4031442) | 100 | 1e-3 | 4×10 | 6.309 | 1.935 | 1.768 | 4.343 | 1.980 | 1.922 | 0.92 | 0.91 | 7.204 | 2.831 | 2.689 | 5.283 | 2.841 | 0.95 | 1.24 | 1.000/1.000 | auto-pw:16 |
| kathleen-large | W3 (N = 4031442) | 100 | 1e-6 | 1×40 | 37.033 | 28.431 | 7.352 | 7.283 | 28.107 | 7.283 | 1.01 | 0.26 | 39.752 | 31.242 | 9.825 | 9.818 | 30.798 | 0.31 | 1.32 | 1.010/0.984 | planewave |
| kathleen-large | W3 (N = 4031442) | 100 | 1e-6 | 4×10 | 39.004 | 32.640 | 7.649 | 7.665 | 29.640 | 7.665 | 1.00 | 0.23 | 40.812 | 34.589 | 9.740 | 9.659 | 31.421 | 0.28 | 1.25 | 1.010/0.984 | planewave |

#### The rule's levels above 1.10× the fastest

| machine | workload | kD | ε | split | level | p | V pairs | auto-pw ran | fastest | ratio | lost s | lost ÷ case best |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| kathleen | W1 | 10 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 9.31 | 0.0043 | 8.313 |
| kathleen | W1 | 30 | 1e-3 | 1×40 | 2 | 12 | 3096 | pw i2 | rot. | 1.86 | 0.0018 | 0.230 |
| kathleen | W1 | 30 | 1e-3 | 1×40 | 3 | 8 | 32256 | b-f | rot. | 1.50 | 0.0029 | 0.368 |
| kathleen | W3 (N = 108849) | 30 | 1e-3 | 1×40 | 2 | 12 | 3096 | pw i2 | rot. | 2.16 | 0.0019 | 0.058 |
| kathleen | W3 (N = 108849) | 30 | 1e-3 | 1×40 | 3 | 8 | 53352 | b-f | rot. | 1.23 | 0.0019 | 0.059 |
| kathleen | W3 (N = 4032) | 10 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 8.55 | 0.0044 | 3.117 |
| kathleen | W3 (N = 4032) | 10 | 1e-3 | 1×40 | 3 | 7 | 5544 | b-f | rot. | 3.18 | 0.0018 | 1.278 |
| kathleen | W4 clusters (N = 100000) | 1 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 9.03 | 0.0042 | 0.076 |
| kathleen | W4 clusters (N = 100000) | 1 | 1e-3 | 1×40 | 3 | 7 | 19782 | b-f | rot. | 2.15 | 0.0028 | 0.050 |
| kathleen | W4 clusters (N = 100000) | 1 | 1e-3 | 1×40 | 4 | 7 | 43392 | b-f | rot. | 1.14 | 0.0007 | 0.013 |
| kathleen | W4 clusters (N = 100000) | 100 | 1e-3 | 1×40 | 2 | 21 | 3096 | pw i2 | rot. | 1.15 | 0.0011 | 0.016 |
| kathleen | W4 clusters (N = 100000) | 100 | 1e-3 | 1×40 | 3 | 13 | 19782 | pw i2 | rot. | 1.33 | 0.0034 | 0.049 |
| kathleen | W4 clusters (N = 100000) | 100 | 1e-3 | 1×40 | 4 | 8 | 43392 | b-f | rot. | 1.36 | 0.0023 | 0.034 |
| kathleen | W4 cube (N = 100000) | 1 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 8.61 | 0.0040 | 0.139 |
| kathleen | W4 plummer (N = 100000) | 1 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 9.24 | 0.0043 | 0.089 |
| kathleen | W4 plummer (N = 100000) | 1 | 1e-3 | 1×40 | 8 | 7 | 35148 | b-f | rot. | 1.33 | 0.0014 | 0.029 |
| kathleen | W4 plummer (N = 100000) | 100 | 1e-3 | 1×40 | 8 | 7 | 35148 | b-f | rot. | 1.33 | 0.0016 | 0.009 |
| kathleen | W4 sphere (N = 100000) | 1 | 1e-3 | 1×40 | 2 | 7 | 3096 | b-f | rot. | 8.54 | 0.0042 | 0.114 |
| kathleen-large | W3 (N = 4031442) | 100 | 1e-3 | 1×40 | 2 | 30 | 3096 | pw i2 | rot. | 1.19 | 0.0037 | 0.002 |

### locust's first pass (a record: beside another session's CUDA compile)

#### Summary

| machine | workload | kD | ε | split | M2L Σ rotation | auto | auto-pw | planewave | best per level | auto-pw/best | auto-pw/auto | evaluate rotation | auto | auto-pw | planewave | auto-pw/auto | ABBA max | auto-pw ÷ rot. error φ/∇φ | auto-pw's checksum also |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| locust-first | W1 | 10 | 1e-3 | 1×72 | 0.000 | 0.004 | 0.004 | 0.001 | 0.000 | 23.05 | 0.99 | 0.001 | 0.005 | 0.005 | 0.002 | 1.00 | 1.15 | 1.000/1.000 | auto |
| locust-first | W1 | 10 | 1e-6 | 1×72 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.07 | 1.07 | 0.003 | 0.002 | 0.002 | 0.002 | 0.98 | 1.21 | 1.005/0.995 | planewave |
| locust-first | W1 | 30 | 1e-3 | 1×72 | 0.002 | 0.009 | 0.009 | 0.003 | 0.002 | 5.58 | 1.06 | 0.005 | 0.012 | 0.012 | 0.006 | 1.03 | 1.07 | 1.000/1.000 | – |
| locust-first | W1 | 30 | 1e-6 | 1×72 | 0.007 | 0.006 | 0.004 | 0.005 | 0.005 | 0.91 | 0.68 | 0.010 | 0.010 | 0.008 | 0.009 | 0.80 | 1.28 | 1.000/1.000 | planewave |
| locust-first | W1 | 100 | 1e-3 | 1×72 | 0.055 | 0.055 | 0.042 | 0.052 | 0.042 | 1.00 | 0.77 | 0.087 | 0.087 | 0.076 | 0.084 | 0.87 | 1.11 | 1.000/1.000 | – |
| locust-first | W1 | 100 | 1e-6 | 1×72 | 0.135 | 0.135 | 0.070 | 0.070 | 0.070 | 1.00 | 0.52 | 0.203 | 0.202 | 0.138 | 0.137 | 0.68 | 1.06 | 0.995/0.980 | planewave |
| locust-first | W1 | 500 | 1e-3 | 1×72 | 5.468 | 5.288 | 3.203 | 3.463 | 3.244 | 0.99 | 0.61 | 6.068 | 5.889 | 3.796 | 4.047 | 0.64 | 1.17 | 1.000/1.000 | – |
| locust-first | W1 | 500 | 1e-6 | 1×72 | 9.738 | 9.692 | 4.368 | 4.355 | 4.363 | 1.00 | 0.45 | 10.535 | 10.490 | 5.148 | 5.153 | 0.49 | 1.07 | 0.973/0.984 | planewave |
| locust-first | W3 (N = 108849) | 30 | 1e-3 | 1×72 | 0.018 | 0.025 | 0.026 | 0.021 | 0.016 | 1.65 | 1.03 | 0.042 | 0.049 | 0.050 | 0.045 | 1.01 | 1.40 | 1.000/1.000 | – |
| locust-first | W3 (N = 108849) | 30 | 1e-6 | 1×72 | 0.104 | 0.104 | 0.031 | 0.031 | 0.031 | 1.00 | 0.30 | 0.135 | 0.134 | 0.063 | 0.062 | 0.47 | 1.07 | 0.973/0.964 | planewave |
| locust-first | W3 (N = 4031442) | 100 | 1e-3 | 1×72 | 1.405 | 0.892 | 0.898 | 1.301 | 0.871 | 1.03 | 1.01 | 2.143 | 1.624 | 1.617 | 2.016 | 1.00 | 1.10 | 1.000/1.000 | – |
| locust-first | W3 (N = 4031442) | 100 | 1e-6 | 1×72 | 8.551 | 8.477 | 1.766 | 1.740 | 1.740 | 1.01 | 0.21 | 9.587 | 9.486 | 2.752 | 2.740 | 0.29 | 1.19 | 1.010/0.984 | planewave |
| locust-first | W3 (N = 4032) | 10 | 1e-3 | 1×72 | 0.000 | 0.007 | 0.007 | 0.001 | 0.000 | 18.05 | 1.00 | 0.008 | 0.014 | 0.014 | 0.008 | 1.00 | 1.05 | 1.000/1.000 | auto |
| locust-first | W3 (N = 4032) | 10 | 1e-6 | 1×72 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 1.02 | 0.95 | 0.019 | 0.019 | 0.019 | 0.019 | 0.99 | 1.09 | 0.991/0.947 | planewave |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 0.028 | 0.057 | 0.057 | 0.036 | 0.028 | 2.02 | 1.00 | 0.087 | 0.117 | 0.118 | 0.096 | 1.00 | 1.17 | 1.000/1.000 | auto |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-6 | 1×72 | 0.174 | 0.174 | 0.059 | 0.060 | 0.060 | 0.99 | 0.34 | 0.324 | 0.321 | 0.211 | 0.206 | 0.66 | 1.12 | 0.969/0.916 | planewave |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 0.033 | 0.057 | 0.057 | 0.042 | 0.032 | 1.77 | 1.01 | 0.095 | 0.118 | 0.119 | 0.104 | 1.00 | 1.24 | 1.000/1.000 | – |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-6 | 1×72 | 0.178 | 0.178 | 0.067 | 0.068 | 0.068 | 0.99 | 0.38 | 0.326 | 0.331 | 0.215 | 0.218 | 0.65 | 1.06 | 0.970/0.921 | planewave |
| locust-first | W4 cube (N = 100000) | 1 | 1e-3 | 1×72 | 0.017 | 0.024 | 0.024 | 0.017 | 0.015 | 1.61 | 0.99 | 0.037 | 0.046 | 0.045 | 0.039 | 0.99 | 1.16 | 1.000/1.000 | auto |
| locust-first | W4 cube (N = 100000) | 1 | 1e-6 | 1×72 | 0.104 | 0.104 | 0.028 | 0.029 | 0.029 | 0.97 | 0.27 | 0.132 | 0.133 | 0.057 | 0.058 | 0.43 | 1.36 | 0.983/0.928 | planewave |
| locust-first | W4 cube (N = 100000) | 100 | 1e-3 | 1×72 | 0.063 | 0.063 | 0.041 | 0.042 | 0.042 | 0.99 | 0.66 | 0.089 | 0.090 | 0.068 | 0.068 | 0.76 | 1.10 | 1.000/1.000 | planewave |
| locust-first | W4 cube (N = 100000) | 100 | 1e-6 | 1×72 | 0.129 | 0.128 | 0.049 | 0.051 | 0.051 | 0.97 | 0.38 | 0.161 | 0.160 | 0.081 | 0.083 | 0.51 | 1.21 | 1.003/0.990 | planewave |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 0.022 | 0.060 | 0.060 | 0.027 | 0.021 | 2.84 | 1.00 | 0.088 | 0.126 | 0.127 | 0.095 | 1.01 | 1.11 | 1.000/1.000 | auto |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-6 | 1×72 | 0.133 | 0.133 | 0.040 | 0.040 | 0.040 | 1.01 | 0.30 | 0.296 | 0.302 | 0.205 | 0.204 | 0.68 | 3.02 | 1.063/0.940 | planewave |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 0.068 | 0.090 | 0.076 | 0.057 | 0.051 | 1.49 | 0.84 | 0.136 | 0.159 | 0.146 | 0.123 | 0.92 | 1.11 | 1.000/1.000 | – |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-6 | 1×72 | 0.209 | 0.209 | 0.075 | 0.075 | 0.075 | 1.01 | 0.36 | 0.365 | 0.368 | 0.231 | 0.230 | 0.63 | 1.35 | 1.013/0.955 | planewave |
| locust-first | W4 sphere (N = 100000) | 1 | 1e-3 | 1×72 | 0.021 | 0.028 | 0.028 | 0.028 | 0.019 | 1.49 | 0.99 | 0.051 | 0.060 | 0.058 | 0.060 | 0.97 | 1.09 | 1.000/1.000 | auto |
| locust-first | W4 sphere (N = 100000) | 1 | 1e-6 | 1×72 | 0.125 | 0.125 | 0.049 | 0.049 | 0.049 | 1.00 | 0.39 | 0.192 | 0.193 | 0.114 | 0.115 | 0.59 | 2.21 | 0.986/0.945 | planewave |
| locust-first | W4 sphere (N = 100000) | 100 | 1e-3 | 1×72 | 0.057 | 0.056 | 0.043 | 0.054 | 0.043 | 1.00 | 0.77 | 0.092 | 0.090 | 0.078 | 0.088 | 0.86 | 1.25 | 1.000/1.000 | – |
| locust-first | W4 sphere (N = 100000) | 100 | 1e-6 | 1×72 | 0.149 | 0.151 | 0.073 | 0.074 | 0.074 | 0.99 | 0.49 | 0.218 | 0.223 | 0.142 | 0.143 | 0.64 | 1.04 | 1.003/0.978 | planewave |

#### The rule's levels above 1.10× the fastest

| machine | workload | kD | ε | split | level | p | V pairs | auto-pw ran | fastest | ratio | lost s | lost ÷ case best |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| locust-first | W1 | 10 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 23.36 | 0.0036 | 22.359 |
| locust-first | W1 | 30 | 1e-3 | 1×72 | 2 | 12 | 3096 | pw i2 | rot. | 2.00 | 0.0004 | 0.234 |
| locust-first | W1 | 30 | 1e-3 | 1×72 | 3 | 8 | 32256 | b-f | rot. | 6.59 | 0.0072 | 4.286 |
| locust-first | W3 (N = 108849) | 30 | 1e-3 | 1×72 | 2 | 12 | 3096 | pw i2 | rot. | 2.06 | 0.0004 | 0.027 |
| locust-first | W3 (N = 108849) | 30 | 1e-3 | 1×72 | 3 | 8 | 53352 | b-f | rot. | 5.09 | 0.0094 | 0.604 |
| locust-first | W3 (N = 108849) | 30 | 1e-6 | 1×72 | 2 | 16 | 3096 | pw i2 | rot. | 1.10 | 0.0001 | 0.003 |
| locust-first | W3 (N = 4032) | 10 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 27.33 | 0.0037 | 10.141 |
| locust-first | W3 (N = 4032) | 10 | 1e-3 | 1×72 | 3 | 7 | 5544 | b-f | rot. | 12.16 | 0.0025 | 6.860 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 24.86 | 0.0041 | 0.146 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 3 | 7 | 19782 | b-f | rot. | 7.54 | 0.0046 | 0.164 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 4 | 7 | 43392 | b-f | rot. | 5.32 | 0.0056 | 0.201 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 5 | 7 | 106500 | b-f | rot. | 2.98 | 0.0059 | 0.211 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 6 | 7 | 214818 | b-f | rot. | 1.67 | 0.0040 | 0.142 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 7 | 7 | 289194 | b-f | rot. | 1.44 | 0.0035 | 0.124 |
| locust-first | W4 clusters (N = 100000) | 1 | 1e-3 | 1×72 | 8 | 7 | 351936 | b-f | rot. | 1.12 | 0.0011 | 0.038 |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 4 | 8 | 43392 | b-f | rot. | 6.41 | 0.0096 | 0.297 |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 5 | 7 | 106500 | b-f | rot. | 2.98 | 0.0059 | 0.182 |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 6 | 7 | 214818 | b-f | rot. | 1.66 | 0.0040 | 0.123 |
| locust-first | W4 clusters (N = 100000) | 100 | 1e-3 | 1×72 | 7 | 7 | 289194 | b-f | rot. | 1.41 | 0.0033 | 0.103 |
| locust-first | W4 cube (N = 100000) | 1 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 25.87 | 0.0039 | 0.267 |
| locust-first | W4 cube (N = 100000) | 1 | 1e-3 | 1×72 | 3 | 7 | 53352 | b-f | rot. | 4.21 | 0.0054 | 0.367 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 28.49 | 0.0040 | 0.191 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 3 | 7 | 53352 | b-f | rot. | 4.29 | 0.0054 | 0.257 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 4 | 7 | 53352 | b-f | rot. | 4.25 | 0.0055 | 0.258 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 5 | 7 | 109548 | b-f | rot. | 3.38 | 0.0073 | 0.344 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 6 | 7 | 205458 | b-f | rot. | 2.11 | 0.0063 | 0.296 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 7 | 7 | 333330 | b-f | pw i2 | 1.49 | 0.0038 | 0.181 |
| locust-first | W4 plummer (N = 100000) | 1 | 1e-3 | 1×72 | 8 | 7 | 35148 | b-f | rot. | 6.84 | 0.0066 | 0.311 |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 5 | 7 | 109548 | b-f | rot. | 3.11 | 0.0071 | 0.138 |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 6 | 7 | 205458 | b-f | rot. | 2.04 | 0.0061 | 0.118 |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 7 | 7 | 333330 | b-f | pw i2 | 1.40 | 0.0035 | 0.068 |
| locust-first | W4 plummer (N = 100000) | 100 | 1e-3 | 1×72 | 8 | 7 | 35148 | b-f | rot. | 6.85 | 0.0067 | 0.130 |
| locust-first | W4 sphere (N = 100000) | 1 | 1e-3 | 1×72 | 2 | 7 | 3096 | b-f | rot. | 28.10 | 0.0040 | 0.213 |
| locust-first | W4 sphere (N = 100000) | 1 | 1e-3 | 1×72 | 3 | 7 | 53352 | b-f | rot. | 4.27 | 0.0054 | 0.286 |
