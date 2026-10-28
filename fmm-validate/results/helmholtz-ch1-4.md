# Helmholtz CH1.4: FMM3D convention check and fixtures

Helmholtz task CH1.4 ([docs/helmholtz/tasks/CH1.4.md](../../docs/helmholtz/tasks/CH1.4.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH1.4, §9.2 and R12. Measured on 2026-10-28 from branch `tbetcke/h-ch1-4` against
FMM3D v2.1.0 (commit `70455e0`, T9's pin, built by `tools/fmm3d/build.sh`, variant
`default`): on the M3 Max with Homebrew GCC 16.2.0, and on Kathleen with the gcc 12.3.0
module. Kathleen ran the check and the fixtures in job 614095 (node-c11a-089), and R12 in
job 614099 (one node, 40 threads).

## Summary

- **The draft §3.15** ([docs/helmholtz/conventions-3.15-draft.md](../../docs/helmholtz/conventions-3.15-draft.md))
  covers items 1–9 of proposal §4, drawn from the proposal and the S1–S3 sign-offs. Item 1
  is fixed by this task. CH1.1 asks for sign-off on the items marked proposed, and the
  items marked open name the later task that decides them.
- **The two-point check identifies FMM3D's conventions, and they are the draft's.** Both
  `hfmm3d_s_c_g` and `h3ddirectcg`, on both machines, compute
  φ(x) = Σ q_j e^{ik|x−y_j|}/(4π|x−y_j|) and its gradient in the target x. They use the
  charge as given (not conjugated), take k as given (e^{+ikr}, the outgoing branch for
  Im k ≥ 0) and include the factor 1/(4π). This holds for k = 1, 2.5, 1 + 0.5i, 2.5 + i,
  0.5i and 3i, to at most 2.6·10⁻¹⁶. Exactly one of the 16 candidate conventions matches
  each case. The draft needs no correction.
- **The fixtures** (`fmm-ref/fixtures/helmholtz_fmm3d.json`, 687 KB) are FMM3D's direct
  sum `h3ddirectcg` with 200 sources and 40 targets. They cover real, complex and
  imaginary k (two of each) and real and complex charges: 12 cases. Every value agrees
  with the draft's kernel summed in mpmath at 30 digits to at most 1.0·10⁻¹⁵ relative ℓ2,
  and the worst point is within 2.7·10⁻¹⁵ of its sum of term magnitudes. Kathleen's FMM3D
  reproduces them to 1.7·10⁻¹⁶. CH1.3 tests `nd-fmm-ref`'s direct sum against them.
- **R12: FMM3D is not a reliable reference for purely imaginary zk at tight eps.** For
  κD ≤ 30, `hfmm3d` meets its eps at every eps from 10⁻³ to 10⁻¹², as it does for real k.
  For κD from 50 to 300 it misses eps 10⁻⁹ or 10⁻¹² in most cells (cube 9 of 12, sphere
  10 of 12), by up to 41× at 10⁻⁹ and up to 12,000× at 10⁻¹² (cube, κD = 150:
  1.2·10⁻⁸ at eps 10⁻¹²). At eps 10⁻⁶ it
  is within 1.3× everywhere. At κD = 1000 the potential is all near field and every eps
  passes. The misses are not specific to k = iκ: complex k with Im kD = 100 (Re kD from 1
  to 100) miss in the same way, while k = 100 + 30i passes. They begin where FMM3D's decay
  cut-off leaves an M2L level with Im(k) w ≳ 12 active (§4). Question 2 asks what this
  means for CH2.5 and CH5.3.

## 1. The draft §3.15

Written first, as the roadmap asks. Item 1, the kernel, its branch, its factors and the
gradient, is what this task checks. Items 2–6 restate the signed-off results of S1–S3:

- the real orthonormal basis (cos in slot +m, sin in slot −m, no Condon–Shortley phase);
- the scale s_l = min(|k| a_l, 1) with the circumradius;
- the exponential factor on every level;
- the GD rotation blocks and the β/π − β rule;
- the coaxial symmetries and stored normalisation.

Items 7 and 8 are left open to CH0.6, CH0.9, CH4.1 and D3. Item 9 is proposed. One
choice is new and left to CH1.1: where the factor ik of the expansion is applied
(draft item 1, "Where the factors go").

## 2. The two-point check

`fmm3d_driver helmholtz-check` (`tools/fmm3d/driver.f90`) places q₁ = 0.3 − 0.7i at
x₁ = (0, 0, 0) and q₂ = 1 at x₂ = (1, 0, 0), and compares the four values φ(x₁),
∂ₓφ(x₁), φ(x₂) and ∂ₓφ(x₂) with the draft's values. With E = e^{ik}/(4π), the draft
gives q₂E, q₂(1 − ik)E, q₁E and q₁(ik − 1)E, and ∂_y φ = ∂_z φ = 0. The driver stops
unless they agree to 10⁻¹². Every Helmholtz run of the driver checks this first. A
non-real q₁ separates q from conj q, and the sign of the gradient separates the target
from the source.

`tools/fmm3d/helmholtz.py check` then matches each set of values against 16 candidates:

- the factor, 1 or 1/(4π);
- the kernel, e^{+ikr} or e^{−ikr};
- the gradient, in the target or in the source;
- the charge, q or conj q.

| k | `hfmm3d_s_c_g` (eps 10⁻¹²) | `h3ddirectcg` | candidates matched |
| --- | --- | --- | --- |
| 1 | 1.7·10⁻¹⁶ | 1.7·10⁻¹⁶ | 1: factor 1/(4π), e^{+ikr}, gradient in the target, charge q |
| 2.5 | 1.7·10⁻¹⁶ | 1.7·10⁻¹⁶ | the same |
| 1 + 0.5i | 1.0·10⁻¹⁶ | 1.0·10⁻¹⁶ | the same |
| 2.5 + i | 4.9·10⁻¹⁷ (Kathleen 1.6·10⁻¹⁶) | the same | the same |
| 0.5i | 2.6·10⁻¹⁶ | 2.6·10⁻¹⁶ | the same |
| 3i | 0 | 0 | the same |
| 1 − 0.5i (outside the branch, record only) | 1.4·10⁻¹⁶ | 1.4·10⁻¹⁶ | the same |
| −1 (outside the branch, record only) | 1.7·10⁻¹⁶ | 1.7·10⁻¹⁶ | the same |

The table gives the largest relative difference from the draft, on the M3 Max. Kathleen
is the same to the last digit except where noted. The transverse components ∂_y φ and
∂_z φ are exactly 0 throughout.

- **Factor:** 1/(4π) is included, as in FMM3D's Laplace v2.1.0 (T9).
- **Sign of k:** e^{+ikr}. For real k the conjugate candidate fails by O(1).
- **Gradient:** in the target, ∇ₓ.
- **Branch:** FMM3D takes zk as given. For Im k > 0 it computes the decaying kernel, and
  for k = iκ the real e^{−κr}/(4πr) (φ is real to the last bit for real charges).
- **Charges:** taken as given, not conjugated.

On two points `hfmm3d` evaluates the pair directly (one box), so the check identifies the
interface's conventions. Its far field agrees with its own direct sum within eps for
real and complex k (§4), which shows that the expansions use the same conventions.

FMM3D's FMM drops pairs closer than 2⁻⁵¹ times the root box (`hfmm3d.f`), and its direct
routines drop pairs closer than the caller's `thresh`. The fixtures pass 10⁻³⁰⁰, so only
exact coincidences are dropped, as in CONVENTIONS §3.13. Outside the draft's branch
(Im k < 0, Re k < 0) FMM3D still computes e^{ikr} on two points. The draft rejects such
k in the builder, so nothing depends on this.

## 3. The fixtures

`tools/fixtures/gen_helmholtz_fmm3d.py` (PEP 723, mpmath 1.4.1) makes them:

1. It runs the check and the identification, and fails unless exactly one candidate
   matches each checked case.
2. It writes the problems (seed 20261028):
   - 200 sources uniform in [0, 1]³ and 40 targets in [−0.5, 1.5]³;
   - real charges uniform in [−1, 1], and complex charges with each part uniform in
     [−1, 1];
   - k = 1, 25, 1 + 0.5i, 25 + 2.5i, i and 25i.
3. It runs `h3ddirectcg` through the driver. The values are taken at the 200 sources,
   with the self term dropped, and at the 40 targets.
4. It checks every value against the draft's kernel and gradient summed in mpmath at 30
   digits, and fails above 10⁻¹³.

The header records:

- FMM3D's release, commit and tarball hash;
- the routine and `thresh`;
- the `eps`: none, since this is a direct sum (`hfmm3d` appears only in the two-point
  record);
- the compiler and the machine;
- the agreement of each case with the reference.

| case | rel ℓ2 φ | rel ℓ2 ∇φ | max error / term sum |
| --- | --- | --- | --- |
| k = 1, real q | 4.9·10⁻¹⁶ | 4.4·10⁻¹⁶ | 3.5·10⁻¹⁶ |
| k = 1, complex q | 5.8·10⁻¹⁶ | 4.7·10⁻¹⁶ | 3.5·10⁻¹⁶ |
| k = 25, real q | 1.0·10⁻¹⁵ | 9.9·10⁻¹⁶ | 2.3·10⁻¹⁶ |
| k = 25, complex q | 1.0·10⁻¹⁵ | 9.9·10⁻¹⁶ | 2.8·10⁻¹⁶ |
| k = 1 + 0.5i, real q | 5.6·10⁻¹⁶ | 6.1·10⁻¹⁶ | 4.5·10⁻¹⁶ |
| k = 1 + 0.5i, complex q | 5.3·10⁻¹⁶ | 4.1·10⁻¹⁶ | 2.6·10⁻¹⁶ |
| k = 25 + 2.5i, real q | 7.0·10⁻¹⁶ | 6.0·10⁻¹⁶ | 2.6·10⁻¹⁶ |
| k = 25 + 2.5i, complex q | 6.2·10⁻¹⁶ | 5.6·10⁻¹⁶ | 3.5·10⁻¹⁶ |
| k = i, real q | 5.2·10⁻¹⁶ | 4.9·10⁻¹⁶ | 6.6·10⁻¹⁶ |
| k = i, complex q | 5.8·10⁻¹⁶ | 4.8·10⁻¹⁶ | 3.7·10⁻¹⁶ |
| k = 25i, real q | 5.6·10⁻¹⁶ | 7.1·10⁻¹⁶ | 2.7·10⁻¹⁵ |
| k = 25i, complex q | 4.2·10⁻¹⁶ | 4.3·10⁻¹⁶ | 1.5·10⁻¹⁵ |

Reproducibility:

- On the M3 Max, `--check` regenerates the file byte for byte at 1 and at 8 OpenMP
  threads. `h3ddirectcg` parallelises over targets and sums each target serially.
- On Kathleen (gcc 12.3.0, job 614095), `--check --no-reference` matches every case to
  at most 1.7·10⁻¹⁶ in potential and 1.3·10⁻¹⁶ in gradient. For k = 25i the files are
  identical. The bits differ between compilers, so CH1.3 compares to round-off, not bit
  for bit.

## 4. R12: `hfmm3d` for purely imaginary zk

`tools/fmm3d/helmholtz.py r12` compares `hfmm3d_s_c_g` against `h3ddirectcg` at 1,000
sampled sources, using the relative ℓ2 errors of φ and ∇φ of proposal §9.1. The
reference is FMM3D's own direct sum, not ours as R12's row in proposal §8 says. Our
direct sum (CH1.3) does not exist yet. `h3ddirectcg` is checked against the draft's
kernel in mpmath through the fixtures (§3), to 10⁻¹⁵. The sweep covers:

- workloads: the unit cube (side D = 1), and the surface of the sphere of diameter 1;
- charges: complex, uniform per part;
- imaginary k: κD = 0.01, 0.1, 1, 10, 30, 50, 70, 100, 150, 200, 300 and 1000;
- controls: real kD = 1, 10, 30, 100, and complex kD = 10 + i and 100 + 10i;
- eps = 10⁻³, 10⁻⁶, 10⁻⁹ and 10⁻¹².

Each cell below is max(err φ, err ∇φ)/eps for the cube and the sphere, written
cube / sphere. A cell above 1 means FMM3D missed its own eps.

**M3 Max, N = 10⁵, 12 threads** (`bench-results/ch1-4/r12-m3max-n100000.md`, ignored by
git):

| kind | kD | eps 10⁻³ | 10⁻⁶ | 10⁻⁹ | 10⁻¹² | s at 10⁻¹² (cube, sphere) |
| --- | --- | --- | --- | --- | --- | --- |
| real | 1 | 0.13 / 0.06 | 0.08 / 0.02 | 0.03 / 0.01 | 0.12 / 0.20 | 1.28, 2.26 |
| real | 10 | 0.06 / 0.04 | 0.03 / 0.01 | 0.02 / 0.01 | 0.11 / 0.19 | 1.45, 2.54 |
| real | 30 | 0.09 / 0.06 | 0.02 / 0.01 | 0.01 / 0.00 | 0.10 / 0.19 | 1.64, 2.89 |
| real | 100 | **2.02 / 1.07** | 0.66 / 0.28 | 0.32 / 0.12 | 0.21 / 0.19 | 2.02, 3.33 |
| imaginary | 0.01 | 0.10 / 0.05 | 0.04 / 0.03 | 0.02 / 0.01 | 0.12 / 0.20 | 1.05, 1.93 |
| imaginary | 0.1 | 0.10 / 0.05 | 0.06 / 0.04 | 0.02 / 0.01 | 0.12 / 0.19 | 1.04, 1.96 |
| imaginary | 1 | 0.08 / 0.05 | 0.06 / 0.03 | 0.02 / 0.01 | 0.11 / 0.19 | 1.05, 1.98 |
| imaginary | 10 | 0.24 / 0.06 | 0.07 / 0.02 | 0.03 / 0.01 | 0.12 / 0.19 | 1.03, 1.92 |
| imaginary | 30 | 0.29 / 0.06 | 0.17 / 0.04 | 0.16 / 0.07 | 0.30 / 0.28 | 0.93, 1.80 |
| imaginary | 50 | 0.15 / 0.05 | 0.86 / 0.14 | **20.4 / 1.90** | **25.1 / 5.80** | 0.82, 1.67 |
| imaginary | 70 | 0.10 / 0.04 | 0.39 / 0.04 | **26.8 / 3.34** | **3,104 / 131** | 0.73, 1.56 |
| imaginary | 100 | 0.14 / 0.03 | **1.28** / 0.17 | **21.8** / 0.54 | **240 / 20.2** | 0.66, 1.44 |
| imaginary | 150 | 0.02 / 0.01 | 0.17 / 0.03 | **40.6 / 2.05** | **12,063 / 278** | 0.55, 1.26 |
| imaginary | 200 | 0.00 / 0.01 | 0.83 / 0.19 | 0.00 / 0.23 | **1,797 / 70.3** | 0.52, 1.09 |
| imaginary | 300 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / **2.04** | 0.13 / **14.4** | 0.51, 0.90 |
| imaginary | 1000 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.16 / 0.20 | 0.93, 1.15 |
| complex | 10 + i | 0.10 / 0.05 | 0.03 / 0.01 | 0.02 / 0.01 | 0.12 / 0.19 | 1.49, 2.57 |
| complex | 100 + 10i | **1.88** / 0.40 | 0.51 / 0.11 | 0.27 / 0.06 | 0.15 / 0.20 | 1.94, 3.26 |

**Kathleen, N = 10⁶, 40 threads** (job 614099, `logs/fmm3d/r12-n1000000.md` under the
Kathleen root):

Pending: Kathleen job 614099 is running; this table follows when it ends.

### Findings

- **Real and complex k (the controls):** FMM3D meets every eps except 10⁻³ at kD = 100,
  where it misses by 2×. Its own error tables are coarsest there.
- **Imaginary k, κD ≤ 30:** every eps is met, as for real k. Up to κD = 30 the time is
  0.54–0.93 times that of real k at the same |k|.
- **Imaginary k, κD from 50 to 300, eps ≤ 10⁻⁹:** eps is missed on the cube in 9 of the
  12 cells (it passes at κD = 200 with 10⁻⁹, and at κD = 300 with 10⁻⁹ and 10⁻¹²) and on
  the sphere in 10 of 12. The miss
  is worst at 10⁻¹², 12,000× on the cube at κD = 150. At eps 10⁻⁶ the worst cell is
  1.28×. At κD = 1000 the potential is almost all near field, so every eps passes.
- **Large Im k in general:** the diagnostic
  (`bench-results/ch1-4/r12-m3max-diagnostic.md`, cube, N = 10⁵) gives error/eps of
  239, 239, 235, 201 and 155 at eps 10⁻¹² for k = 100i, 1 + 100i, 10 + 100i, 50 + 100i
  and 100 + 100i, against 0.11 for k = 100 + 30i. So the misses follow Im k, not Re k = 0.
- **Where they begin:** in `hfmm3d.f`, a level l with box width w_l does no multipole
  work when e^{−Im k · w_l}/w_l < eps. Below Im(k w_l) = 12π (and Re(k w_l) ≤ 16π) its
  M2L uses plane waves from `hwts3e`, and above that point and shoot. At κD = 150, level 3
  (κw = 18.75) is cut off exactly while eps > 5.7·10⁻⁸. The diagnostic finds error/eps
  0.00 at eps 10⁻⁷, 0.81 at 5·10⁻⁸ and 4.05 at 10⁻⁸. At κD = 30 the largest κw of any
  active level is 7.5, and it passes. At κD = 50 and 100, levels with κw = 12.5 fail at
  10⁻⁹. The tree depth also depends on eps: `hndiv.f` puts at most 80, 200, 400 and 600
  points in a leaf at eps 10⁻³, 10⁻⁶, 10⁻⁹ and 10⁻¹². So at N = 10⁵ the cube's tree
  stops at level 3 for 10⁻⁹ and 10⁻¹². At κD = 200 and eps 10⁻⁹ every level is then cut
  off or a leaf, and FMM3D sums everything directly (error 0.00). At 10⁻⁶ and 10⁻¹², a
  level with multipole work remains. The errors therefore start once a level with Im(k) w ≳ 12 is active at tight
  eps. Telling apart its plane-wave quadrature (`hwts3e` for imaginary zk·w) from its
  degree rule (`h3dterms`) would need FMM3D's internals changed, which this task does not
  do. The locus is a hypothesis that fits every cell.

Sampled targets can miss errors confined to a few boxes (proposal §9.2), so this R12
answer stands on the sampled ℓ2 norm only.

## 5. Consequences and questions for sign-off

1. **The draft §3.15 as CH1.1's starting point.** Item 1 matches FMM3D, so no correction
   is needed. CH1.1 asks for sign-off on the items marked proposed, among them where the
   factor ik is applied (draft item 1: the three crossing operators M2L, M2P and P2L, or
   P2P divided by ik).
2. **R12 and CH2.5 (and CH5.3).** The proposal's fallback is: "if it fails, Yukawa keeps
   our direct sum and the k = iκ cross-check only (CH2.5 skipped, the reason recorded)".
   FMM3D fails only in part. Recommendation: CH2.5 runs, comparing against FMM3D where
   FMM3D met its eps here, which is κD ≤ 30 at every eps, and eps ≥ 10⁻⁶ at larger κD.
   There the sampled direct sum is the measure and FMM3D's error is stated beside it.
   Alternatives: skip CH2.5, as the proposal says; or run CH2.5 everywhere and report
   FMM3D's misses as FMM3D's. CH5.3 (complex k) meets the same limit for Im kD ≳ 50 and
   should restrict its grid in the same way.
3. **For the proposal's next draft:** R12's row records the partial result and the
   limit on Im k; §9.2's "whether it is accurate … is not verified" becomes this result.
   The draft §3.15 item 1 states FMM3D's thresh rule beside §3.13's.

## 6. Files

- `docs/helmholtz/conventions-3.15-draft.md`: the draft.
- `tools/fmm3d/driver.f90`: the `helmholtz-check` and `helmholtz` modes. The Laplace
  usage is unchanged: `main`'s driver and this branch's write byte-identical `.out`,
  `.vec.out` and `.full.out` files on `cube-n100000.bin` at eps 10⁻⁶, 4 threads.
- `tools/fmm3d/build.sh`: the step `driver`.
- `tools/fmm3d/helmholtz.py`: the formats, the check, the identification and R12.
- `tools/fixtures/gen_helmholtz_fmm3d.py` and `fmm-ref/fixtures/helmholtz_fmm3d.json`.
- `tools/kathleen/jobs/fmm3d.sbatch`: the steps `driver`, `helmholtz-check`,
  `helmholtz-fixtures` and `r12`.
- `tools/fmm3d/README.md` and `tools/fixtures/README.md`.
