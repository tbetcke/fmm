#!/bin/sh
# tools/kathleen/probe.sh: the Kathleen queue probe (Phase 5N T1, step 0). POSIX sh.
#
#     tools/kathleen/probe.sh drive [rounds [minutes [host]]]         # on the M3 Max
#     ssh kathleen 'sh -s <command>' < tools/kathleen/probe.sh       # one command there
#
# Measures how long Kathleen's queue makes a small multi-node job wait, before anything
# is installed there. Each probe job runs `srun hostname` (one line per node) and
# `sleep 30` on 2 nodes in the `test` QoS, with --time=00:05:00 and
# --ntasks-per-node=1 (whole nodes: Kathleen's nodes are exclusive). No job is larger
# than 2 nodes (docs/phase5n/README.md, "Working on Kathleen"; the first probe on
# 2026-10-10 also submitted one 4-node `small` job, cancelled when the cap was set).
#
# On Kathleen it writes only under ~/Scratch/fmm-probe/ (FMM_PROBE_DIR overrides): the
# job file, the jobs' output, and
#     submissions.txt   one line per job: round, shape, job id, submit time (epoch, ISO)
#     estimates.txt     per submission: `sbatch --test-only`, the queue depth by QoS
#                       (pending and running), `sinfo -s`, and `squeue --start` for
#                       the new job a minute after submitting
#     cancelled.txt     the probe jobs cancelled after 12 hours pending
#     sacct.txt         `sacct -X` for every probe job (written by `collect`)
#
# Commands run on the login node (short and light: submitting and reading):
#     submit        submit one job (skipped when the user already has 2 jobs in the
#                   `test` QoS) and record the estimates
#     cancel-stale  cancel the probe jobs pending for more than 12 hours
#     pending       print the number of probe jobs still in the queue
#     collect       write and print sacct.txt
#     summary       per job: wait (Start - Submit), run time, state; per shape: minimum,
#                   median and maximum wait of the jobs that started
# and on the M3 Max:
#     drive [rounds [minutes [host]]]  submit a job every <minutes> (default 10
#                   rounds, 150 minutes, host kathleen), cancel stale jobs before each,
#                   then wait until no probe job is left in the queue, and print the
#                   summary. The defaults run for about a day; start it in the
#                   background.
#
# The login-node commands use GNU date (RHEL). ssh reads ~/.ssh, which the Claude Code
# sandbox denies: run `drive` outside the sandbox.

set -eu

dir=${FMM_PROBE_DIR:-$HOME/Scratch/fmm-probe}
max_pending=43200 # seconds: cancel a probe job pending for longer than 12 hours

# The job file of one shape: write_job <name> <qos> <nodes>.
write_job() {
    cat >"$dir/$1.sh" <<EOF
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH --partition=kathleen
#SBATCH --qos=$2
#SBATCH --nodes=$3
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=$dir/%x-%j.out
srun hostname
sleep 30
EOF
}

# Appends the scheduler's view to estimates.txt, headed by $1.
record_queue() {
    {
        echo "== $1 $(date -Iseconds)"
        echo "-- sbatch --test-only"
        sbatch --test-only "$dir/fmm-probe-2n.sh" 2>&1 || true
        echo "-- pending jobs by QoS"
        squeue -h -t PD -o %q | sort | uniq -c
        echo "-- running jobs by QoS"
        squeue -h -t R -o %q | sort | uniq -c
        echo "-- sinfo -s"
        sinfo -s
    } >>"$dir/estimates.txt"
}

submit() {
    mkdir -p "$dir"
    write_job fmm-probe-2n test 2
    last=0
    [ ! -s "$dir/submissions.txt" ] || last=$(cut -d ' ' -f 1 "$dir/submissions.txt" | sort -n | tail -n 1)
    round=$((last + 1))
    record_queue "round $round, before submitting"
    in_test=$(squeue -h -u "$USER" -q test -o %i | wc -l)
    if [ "$in_test" -ge 2 ]; then
        echo "round $round: $in_test jobs already in QoS test, nothing submitted" |
            tee -a "$dir/estimates.txt"
        return 0
    fi
    id=$(sbatch --parsable "$dir/fmm-probe-2n.sh")
    now=$(date +%s)
    echo "$round probe-2n $id $now $(date -d "@$now" -Iseconds)" >>"$dir/submissions.txt"
    echo "round $round: submitted probe-2n as $id"
    sleep 60
    {
        echo "-- squeue --start, a minute after submitting"
        squeue --start -j "$id" -o '%.10i %.12j %.6q %.4D %.10T %.20S %.20V %R' 2>&1 || true
    } >>"$dir/estimates.txt"
    tail -n 4 "$dir/estimates.txt"
}

# The ids of the probe jobs still in the queue, one per line.
queued() {
    squeue -h -u "$USER" -o '%i %j %T %V' | awk '$2 ~ /^fmm-probe-/'
}

cancel_stale() {
    now=$(date +%s)
    queued | while read -r id name state submit; do
        [ "$state" = PENDING ] || continue
        age=$((now - $(date -d "$submit" +%s)))
        if [ "$age" -gt "$max_pending" ]; then
            scancel "$id"
            echo "$id $name cancelled after ${age} s pending ($(date -Iseconds))" |
                tee -a "$dir/cancelled.txt"
        fi
    done
}

all_ids() {
    cut -d ' ' -f 3 "$dir/submissions.txt" | paste -sd , -
}

collect() {
    sacct -X -j "$(all_ids)" -P \
        --format=JobID,JobName,QOS,NNodes,Submit,Start,End,Elapsed,State,NodeList \
        >"$dir/sacct.txt"
    column -t -s '|' "$dir/sacct.txt"
}

# Wait and run time per job, then minimum, median and maximum wait per shape (the
# 4-node shape only from the first probe's records).
summary() {
    collect >/dev/null
    tail -n +2 "$dir/sacct.txt" | while IFS='|' read -r id name qos nodes sub start _ el state _; do
        case "$start" in
        Unknown | None | "") wait=- ;;
        *) wait=$(($(date -d "$start" +%s) - $(date -d "$sub" +%s))) ;;
        esac
        echo "$id $name $qos $nodes $sub $wait $el $state"
    done | tee "$dir/waits.txt" | awk 'BEGIN { print "JobID Name QOS Nodes Submit Wait(s) Elapsed State" } { print }' |
        column -t
    echo
    for shape in fmm-probe-2n fmm-probe-4n; do
        awk -v s="$shape" '$2 == s && $6 != "-" { print $6 }' "$dir/waits.txt" | sort -n |
            awk -v s="$shape" '{ w[NR] = $1 }
                END {
                    if (NR == 0) { print s ": no job started"; exit }
                    m = (NR % 2) ? w[(NR + 1) / 2] : (w[NR / 2] + w[NR / 2 + 1]) / 2
                    printf "%s: %d started, wait min %d s, median %d s, max %d s\n", s, NR, w[1], m, w[NR]
                }'
    done
}

drive() {
    rounds=${1:-10}
    minutes=${2:-150}
    host=${3:-kathleen}
    self=$0
    i=1
    while [ "$i" -le "$rounds" ]; do
        ssh "$host" 'sh -s cancel-stale' <"$self"
        ssh "$host" 'sh -s submit' <"$self"
        [ "$i" -eq "$rounds" ] || sleep $((minutes * 60))
        i=$((i + 1))
    done
    while [ "$(ssh "$host" 'sh -s pending' <"$self")" -gt 0 ]; do
        sleep 1800
        ssh "$host" 'sh -s cancel-stale' <"$self"
    done
    ssh "$host" 'sh -s summary' <"$self"
}

case "${1:-}" in
submit) submit ;;
cancel-stale) cancel_stale ;;
pending) queued | wc -l | tr -d ' ' ;;
collect) collect ;;
summary) summary ;;
drive) shift; drive "$@" ;;
*)
    echo "usage: probe.sh submit|cancel-stale|pending|collect|summary|drive [rounds [minutes [host]]]" >&2
    exit 2
    ;;
esac
