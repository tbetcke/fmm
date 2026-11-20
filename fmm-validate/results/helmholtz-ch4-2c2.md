# Helmholtz CH4.2c2: the interpolated transfer build in the operator, and W1's tables timed

Task: `docs/helmholtz/tasks/CH4.2c2.md`. Roadmap: H4, CH4.2c2 (priority list item 10), the second
half of CH4.2c. Done when: "the operator's build setting (the rule's constants by the build:
C_T = 0.23 interpolated) and the harness's; the FMM within 1.1× Rotation's errors with it; W1 at
kD = 500 and 1000's diagonal tables timed at 1 × 40 and 40 × 1 against CH4.2b2's (2.3 s and 61 s
at p = 131), with the errors and the classes per level against the direct build's; which build
is the default, for sign-off."

## 0. Summary

- **The setting.** `TransferBuild::{Direct, Interpolated}` is now a setting of the operator
  (`HelmholtzOperator::with_diagonal_build`) and of the builder
  (`HelmholtzFmmBuilder::diagonal_build`). The rule reads the build's constants: C_T = 0.15 for
  direct, 0.23 for interpolated, and f32's constants for f32 by either build. The harness takes
  `--diagonal-build direct|interpolated`. Since question 1's sign-off the default is
  `Interpolated` (`DEFAULT_DIAGONAL_BUILD`); `Direct` is named.
- **The errors** (§2). The interpolated build's FMM is 1.000× Rotation's errors for φ and ∇φ on
  all three W1 cases at both splits. That is the same as the direct build's. CI's tests hold it
  within 1.1× on the cube at kD = 40 + 2i and the clusters at kD = 20.
- **The classes** (§2). Each level takes the same classes under both builds on every case,
  10⁻⁸ included. The looser C_T removed no class here. The operator's unit test shows that it
  can remove one: at k = 40, 10⁻⁴, two near classes of level 2 drop out.
- **The tables' build** (§3). At p = 131 (W1 at kD = 500, level 2), CH4.2b2 measured 2.3 s at
  1 × 40 and 61 s at 40 × 1:
  - the direct build, now split by ring, takes 1.80 s and 53.8 s;
  - the interpolated build takes 0.33 s and 6.84 s.

  Over all of a run's diagonal levels, the interpolated build is 4.0–5.7× faster at 1 × 40 and
  3.3–6.4× faster at 40 × 1. At kD = 1000 and 1 × 40 the tables drop from 11.5 s to 2.0 s. At
  kD = 500 and 40 × 1 they drop from 65.4 s to 10.2 s.
- **Bits** (§4). Each build gives one checksum at 1 × 40 and at 40 × 1. The direct build's
  checksum at kD = 500 and 10⁻⁶ is CH4.2b2's `c74ca0e7954cf9ba`, so the direct tables split by
  ring are CH4.2b2's bits in the full FMM.
- **Sign-off** (§5): both questions as recommended; `Interpolated` is the default build.

## 1. Setup

- **Job.** Kathleen job 653605 on node-c11a-058 (`singlenode`, 2 × 20 Cascade Lake cores), from
  13:42 to 14:01 on 2026-11-20. It ran the release build, then
  `tools/helmholtz/ch4-2c2-runs.sh logs/ch4-2c2 node cores`. BLAS threads were 1. The 1 × 40
  runs were one rank bound to every core. The 40 × 1 runs placed 20 ranks per socket with
  `--map-by ppr:20:socket:PE=1 --bind-to core`.
- **Runs.** Each launch uses `helmholtz_accuracy` on W1 (§9.1: complex charges, 1,000 samples,
  gradients) with `--near-v cost --near-wx cost --levels --evaluations 2 --m2l rotation
  --diagonal off,rule --diagonal-build direct,interpolated`. That is three FMMs per launch:
  Rotation, Rotation with the direct diagonal form, and Rotation with the interpolated form.
  Each FMM is evaluated twice; the rows report the second evaluation. The cases are CH4.2b2's:
  kD = 500 at 10⁻⁶ and 10⁻⁸, and kD = 1000 at 10⁻⁶.
- **Tables.** `tools/helmholtz/ch4-2c2-tables.py` makes them from the outputs. The raw outputs
  are in `~/projects/fmm-archive/kathleen/runs/ch4-2c2/`.
- **Seconds.** "Diagonal tables" is the build part of that name (`TableTimings::diagonal`: the
  rule, the transfer functions and the conversions of every level). "Transfer build" is one
  level's transfer functions (`diagonal_build_time`). Both are the slowest rank's.

## 2. Errors and classes

The relative ℓ2 errors over the 1,000 samples. "÷ Rotation" is the row's error over the Rotation
row's, for φ and ∇φ.

| kD | ε | split | Rotation φ / ∇φ | direct ÷ Rotation | interpolated ÷ Rotation | levels:classes (both builds) |
| ---: | --- | --- | --- | --- | --- | --- |
| 500 | 10⁻⁶ | 1 × 40 | 1.18e-7 / 2.75e-7 | 1.000/1.000 | 1.000/1.000 | 2–6: 34 each |
| | | 40 × 1 | 1.18e-7 / 2.75e-7 | 1.000/1.000 | 1.000/1.000 | 2–6: 34 each |
| 500 | 10⁻⁸ | 1 × 40 | 2.72e-9 / 6.32e-9 | 1.000/1.000 | 1.000/1.000 | 2, 3: 34; 4, 5: 32; 6: 12 |
| | | 40 × 1 | 2.72e-9 / 6.32e-9 | 1.000/1.000 | 1.000/1.000 | 2, 3: 34; 4, 5: 32; 6: 12 |
| 1000 | 10⁻⁶ | 1 × 40 | 1.03e-7 / 2.39e-7 | 1.000/1.000 | 1.000/1.000 | 2–7: 34 each |
| | | 40 × 1 | 1.03e-7 / 2.39e-7 | 1.000/1.000 | 1.000/1.000 | 4–7: 34 each (2, 3 on RCR) |

- **The acceptance holds:** the interpolated build is within 1.1× Rotation's errors, at 1.000×.
- **Classes.** The interpolated rule took exactly the direct rule's classes on every level of
  every case.
  - At 10⁻⁸, levels 4–6 are where a class lies near θ ε. The interpolated bound E is larger
    there, but no class crossed θ ε.
  - So on W1 the interpolated build costs no class. Where it could cost one, the operator's
    unit test shows the effect: at k = 40, ε = 10⁻⁴, level 2 (p = 22), the classes [0, 1] stay
    on RCR under the interpolated build (`tests/helmholtz/diagonal.rs`).
- **The large degrees in the FMM.** CH4.2c1 left p ≥ 72 to its ignored pair tests. At E's floor
  those tests read 1.79–3.04 E against RCR, which CH4.2c1 put down to RCR's own floor. W1 at
  kD = 1000 at 1 × 40 runs p = 245, 131 and 72 on levels 2–4 with the interpolated tables, and
  its error is Rotation's to three digits. This is the FMM evidence the `/next-task` rule asks
  for beside CH4.2c1's pair evidence.
- **Not a build effect.** At kD = 1000 and 40 × 1, levels 2 and 3 run RCR under both builds.
  Their scratch at 40 ranks per node exceeds a rank's 205 MiB share of the budget (40 ranks
  over budget at level 2, 15 at level 3), as CH4.1b's census predicted.

## 3. The tables' build

### 3.1 Per level

The transfer functions' build per level, in seconds, the slowest rank's:

| kD, ε | split | level | p | direct | interpolated | direct ÷ interpolated |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 500, 10⁻⁶ | 1 × 40 | 2 | 131 | 1.797 | 0.332 | 5.4 |
| | | 3 | 72 | 0.301 | 0.097 | 3.1 |
| | | 4 | 41 | 0.062 | 0.033 | 1.9 |
| | | 5 | 25 | 0.016 | 0.017 | 0.9 |
| | | 6 | 16 | 0.006 | 0.007 | 0.9 |
| | 40 × 1 | 2 | 131 | 53.826 | 6.844 | 7.9 |
| | | 3 | 72 | 9.125 | 2.105 | 4.3 |
| | | 4 | 41 | 1.788 | 0.712 | 2.5 |
| | | 5 | 25 | 0.462 | 0.280 | 1.7 |
| | | 6 | 16 | 0.142 | 0.124 | 1.1 |
| 500, 10⁻⁸ | 1 × 40 | 2 | 135 | 1.930 | 0.353 | 5.5 |
| | 40 × 1 | 2 | 135 | 58.332 | 7.562 | 7.7 |
| 1000, 10⁻⁶ | 1 × 40 | 2 | 245 | 9.379 | 1.084 | 8.7 |
| | | 3 | 131 | 1.325 | 0.291 | 4.6 |
| | | 4 | 72 | 0.229 | 0.083 | 2.8 |
| | 40 × 1 | 4 | 72 | 9.168 | 2.275 | 4.0 |
| | | 5 | 41 | 1.802 | 0.755 | 2.4 |

- **Against CH4.2b2.** At p = 131, CH4.2b2's direct build by class took 2.34 s at 1 × 40 and
  61.0 s at 40 × 1. Split by ring (CH4.2c1), the direct build takes 1.80 s and 53.8 s. At
  1 × 40 the one-class floor is gone: 34 classes × 66 rings are 2,244 parts on 40 threads. At
  40 × 1 the split gains nothing, since every rank builds every part on its one thread. The
  ring build's 12 % gain there is most likely the node or the run; it was not measured further.
- **Interpolated against direct.** The interpolated build is 5.4× faster at p = 131 and 1 × 40,
  and 7.9× at 40 × 1. At p = 245 and 1 × 40 it is 8.7× faster. CH4.2c1 measured one class on a
  Grace core at 6.1× (p = 131) and 9.6× (p = 245).
- **Small degrees.** At p ≤ 25 the two builds cost the same, about 0.005–0.017 s at 1 × 40. The
  interpolated build's angle table, one part per class, is then about as long as the direct
  samples.

### 3.2 Against the evaluation

Per run: the diagonal tables (the build part) and one evaluation (the second), in seconds.

| kD, ε | split | diagonal tables direct | interpolated | evaluate direct | interpolated | build direct | interpolated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500, 10⁻⁶ | 1 × 40 | 2.270 | 0.574 | 3.06 | 3.12 | 4.90 | 3.23 |
| | 40 × 1 | 65.396 | 10.169 | 3.80 | 3.81 | 69.77 | 14.63 |
| 500, 10⁻⁸ | 1 × 40 | 2.454 | 0.610 | 4.24 | 4.37 | 5.21 | 3.39 |
| | 40 × 1 | 71.873 | 11.301 | 5.61 | 5.63 | 76.81 | 16.09 |
| 1000, 10⁻⁶ | 1 × 40 | 11.504 | 2.019 | 14.07 | 14.13 | 25.04 | 15.39 |
| | 40 × 1 | 11.547 | 3.480 | 52.29 | 52.30 | 52.01 | 44.41 |

- **One rank per node.** With the interpolated build the diagonal tables cost 0.14–0.18 of an
  evaluation, down from 0.58–0.82. W1's build at kD = 1000 drops from 25.0 s to 15.4 s. Rotation
  alone builds in 12.8 s there.
- **One rank per core.** The interpolated tables still cost 2.7× an evaluation at kD = 500
  (10.2 s against 3.8 s), down from 17×. Every rank builds every class. CH8.0 (the tables per
  node) is what removes the factor of the ranks per node.
- **The evaluation is the same under both builds.** At 40 × 1 the rows agree to 0.02 s, since
  the tables have the same layout and the pairs the same work. At 1 × 40 the interpolated rows
  are 0.4–3 % slower on the evaluation, and single levels' M2L differ by up to 0.12 s in
  either direction (kD = 1000, level 3: 1.099 against 1.222 s; kD = 500, 10⁻⁸, level 3: 0.185
  against 0.240 s). Each launch ran the builds one after the other on one node and was not
  repeated, so this is run-to-run spread at 40 threads: no level's M2L work depends on the
  build, and at 40 × 1 the same levels agree within 0.01 s.

## 4. Bits and ranks

| kD, ε | build | checksum at 1 × 40 | at 40 × 1 |
| --- | --- | --- | --- |
| 500, 10⁻⁶ | Rotation | 61ade4ae68af8c25 | 61ade4ae68af8c25 |
| | direct | c74ca0e7954cf9ba | c74ca0e7954cf9ba |
| | interpolated | e984cd7d805ccc04 | e984cd7d805ccc04 |
| 500, 10⁻⁸ | direct | a63ece22cbdee35e | a63ece22cbdee35e |
| | interpolated | 191e9a1b2b7c1050 | 191e9a1b2b7c1050 |
| 1000, 10⁻⁶ | direct | 30294ea6cf2d60a7 | 8c02e69d1184e719 (levels 2, 3 on RCR) |
| | interpolated | 51a0a18f061e7777 | f89f07930b5aee13 (levels 2, 3 on RCR) |

- **Constraint 2.** Each build's output is the same bits at both splits wherever the same levels
  take the form.
- **The ring split in the FMM.** The direct build's checksum at kD = 500, 10⁻⁶ is CH4.2b2's at
  1 × 40, 2 × 20 and 40 × 1 (`c74ca0e7954cf9ba`, CH4.2b2's report §6). So CH4.2c1's split by
  ring gives CH4.2b2's FMM output bit for bit.
- **The two builds differ in their bits**, as their tables do (≤ 4.1 u of the class's largest
  sample per sample; CH4.2c1). The errors agree to three digits (§2).

Rank 0's peak resident size: 5.7 and 6.8 GB at 1 × 40 for kD = 500 (one process with three
FMMs), 17.2 GB at kD = 1000; 1.2–1.3 GB at 40 × 1 for kD = 500, 3.9 GB at kD = 1000.

## 5. Questions for sign-off

Signed off 2026-11-20 by the user on #241, both as recommended: "Both questions as
recommended." The default build is `TransferBuild::Interpolated` (`DEFAULT_DIAGONAL_BUILD`) in
the operator and the builder, applied on the branch before the merge; the runs above named
both builds and are unchanged by it.

1. **The default build.** The interpolated build gave the same errors (1.000× Rotation's) and
   the same classes per level as the direct build on every W1 case. Its tables are 4.0–5.7×
   faster at one rank per node and 3.3–6.4× faster at one rank per core.

   *Recommended:* make `TransferBuild::Interpolated` the default of `HelmholtzFmmBuilder` and of
   the operator, with `Direct` kept as a named setting and as the trusted path of the tests. The
   diagonal form itself stays opt-in until CH4.3. CH4.3 then calibrates C_T and θ with the
   interpolated build as the default.

   *Alternative:* keep `Direct` as the default and let CH4.3 decide. This costs its build times
   in CH4.3's sweeps: 61 s → 10 s per rank at 40 × 1.
2. **The acceptance as read.** The FMM within 1.1× Rotation's errors is read in §9.1's measure
   (relative ℓ2 of φ and ∇φ at the samples) on W1's three cases at both splits. In CI it is
   checked on the cube at kD = 40 + 2i and the clusters at kD = 20, ε = 10⁻⁴. The tables are
   timed against CH4.2b2's per level and per run. The classes per level are compared with the
   direct build's.

   *Recommended:* accept.
