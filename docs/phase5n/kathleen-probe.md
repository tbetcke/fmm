# Phase 5N T1, step 0: the Kathleen queue probe

The probe measured how long Kathleen's queue makes a small multi-node job wait, before
anything was installed there (docs/phase5n/T1-kathleen-environment.md, "Step 0"). It ran
from the M3 Max over `ssh kathleen` with `tools/kathleen/probe.sh`, on Saturday
2026-10-10, and wrote only under `~/Scratch/fmm-probe/`.

**Shortened by the user.** The plan was a pair of jobs every 2.5 hours for a day: one
2-node job in the `test` QoS and one 4-node job in the `small` QoS. After the first
pair the user stopped the repeats and capped every Kathleen job at 2 nodes for now
(README, "Working on Kathleen"). The 4-node job, still pending, was cancelled. This report
therefore rests on **one 2-node job**: it shows that a short 2-node job can start within
minutes, not what the queue does on a weekday or over a day.

## The jobs

Each job ran `srun hostname` and `sleep 30` with `--time=00:05:00` and
`--ntasks-per-node=1` (whole nodes; Kathleen's nodes are exclusive).

From `sacct -X -j 238815,238816` (`probe.sh summary`):

| Job | QoS | Nodes | Submit | Start | Wait | Run | State | Nodes used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 238815 | `test` | 2 | 14:59:23 | 15:01:47 | **2 min 24 s** | 31 s | COMPLETED | node-c11b-[018,031] |
| 238816 | `small` | 4 | 14:59:23 | — | not measured | — | CANCELLED at 15:06:57, after 7 min 34 s pending | — |

Both times are BST on 2026-10-10. The 2-node job's output had one line per node, as
expected (`node-c11b-018.kathleen.ucl.ac.uk`, `node-c11b-031.kathleen.ucl.ac.uk`).

Waits per shape: 2 nodes in `test`, one job, 144 s (minimum, median and maximum alike);
4 nodes in `small`, no job started.

## The scheduler's view at submission (14:59)

| Item | Value |
| --- | --- |
| Nodes (`sinfo -s`) | 188 allocated, 1 idle, 1 other, of 190 |
| Pending jobs by QoS | `small` 1,876, `medium` 200; none in `test`, `singlenode` or `large` |
| Running jobs by QoS | `small` 94 |
| `sbatch --test-only`, 2 nodes `test` | start predicted at 16:15:26 (76 minutes) |
| `sbatch --test-only`, 4 nodes `small` | start predicted at 16:18:09 (79 minutes) |
| `squeue --start`, a minute after submitting | `N/A` for both, reason `Resources` |

The prediction for the 2-node job was far too pessimistic: it started after 2.4 minutes,
not 76. A short job in the `test` QoS, with nobody else waiting in it, was started by the
backfill scheduler as soon as two nodes were free. `--test-only` predictions are not a
useful guide to the wait of a short job here.

## Node-hours

Job 238815: 2 nodes × 31 s = 0.017 node-hours. Job 238816 never ran: 0. Total 0.02
node-hours, against T1's proposed budget of 20 (decision 3). Storage used on Kathleen
after the probe: 92 KiB of the 250 GiB quota (`lquota`).

## What the waits mean for the phase

- **T1's builds and checks** fit the `test` QoS (1 h, ≤ 2 nodes, ≤ 2 jobs) if a job
  stays under an hour; one job that waited minutes is weak evidence but encouraging. A
  build longer than an hour goes to `singlenode` (6 h), whose wait was not measured.
- **T1's MPI tests at 2 × 40 ranks**, and **T2's 2-node runs**, are the same shape as the
  probe job: short 2-node jobs in `test`.
- **T2's longer 2-node sweeps** (more than an hour) need `small`, where about 1,900 jobs
  were pending; their wait was not measured. T2 should batch a sweep into as few
  allocations as fit under an hour in `test` before using `small`.
- **4 nodes and beyond** (Phase 5S): not measured, and not allowed in this phase. A
  repeat of the probe with the 4-node shape comes first when the cap is lifted.

## Recommendation for decision 0

**Kathleen is usable for jobs of at most 2 nodes**, with the caveat that this rests on
one job on a Saturday afternoon: proceed with Step 1, in the `test` QoS where a job fits
in an hour, and record every job's wait (from `sacct`) in T1's and T2's reports, so the
evidence grows with the work.

**Decision 0 (the user, 2026-10-10):** Kathleen is usable; proceed with Step 1, every job
at most 2 nodes.

## Repeating the probe

```sh
ssh kathleen 'sh -s submit' < tools/kathleen/probe.sh     # one 2-node job, with the estimates
ssh kathleen 'sh -s summary' < tools/kathleen/probe.sh    # waits from sacct
tools/kathleen/probe.sh drive 10 150                       # every 2.5 hours, ten times
```

`probe.sh`'s header lists what it records and where.
