# tools/kathleen/jobs/lib.sh: shared by the Slurm job scripts in this directory
# (Phase 5N T1). Bash; sourced, never run.
#
# A job script starts with `#!/bin/bash -l` and its #SBATCH lines, then
#
#     cd "$SLURM_SUBMIT_DIR" && . tools/kathleen/jobs/lib.sh
#
# (tools/kathleen/submit.sh submits from the tree that tools/kathleen/sync.sh copied).
# This file then
#   - sources tools/kathleen/env.sh (module purge and the pinned modules; TMPDIR is
#     /dev/shm/fmm-$USER-$SLURM_JOB_ID, in RAM: the nodes are diskless);
#   - creates that TMPDIR on every node of the job, and at exit prints its size per node
#     and removes it (README.md, "Temporary files"). Slurm resets TMPDIR to /tmp inside
#     every srun step, so processes that srun or mpirun start (MPI ranks) write their
#     temporary files to /tmp (RAM, node-local): at exit the job lists, per node, what
#     this user created in /tmp during the job, and removes it;
#   - restricts the batch shell, and so cargo, rustc and the tests it starts, to one
#     hardware thread per physical core (Slurm gives the batch step all 80 hardware
#     threads of a node; hyperthreading stays unused, docs/phase5n/README.md). `srun`
#     and `mpirun` steps get their own binding;
#   - prints the job's facts: id, QoS, nodes, node list, the revision of the tree, and
#     the environment's summary line;
#   - defines `step <name> <command...>`, which runs a command, times it, records the
#     result and carries on after a failure, and prints the table of every step at exit.
#     The job exits non-zero if any step failed.

set -uo pipefail

. tools/kathleen/env.sh

_fmm_job_start=$(date +%s)
_fmm_job_steps=
_fmm_job_failed=0

# One line of facts per item, at the top of the output.
printf '=== job %s (%s), QoS %s, %s node(s): %s; submitted from %s\n' \
    "$SLURM_JOB_ID" "${SLURM_JOB_NAME:-}" "${SLURM_JOB_QOS:-?}" "$SLURM_JOB_NUM_NODES" \
    "$SLURM_JOB_NODELIST" "$SLURM_SUBMIT_DIR"
printf '=== source revision: %s\n' "$(tr '\n' ' ' <.source-revision 2>/dev/null || echo unknown)"
printf '=== start %s\n' "$(date -Iseconds)"

# Runs a command on every node of the job, once per node.
on_every_node() {
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        --ntasks-per-node=1 --cpu-bind=none "$@"
}

# The directory, and a stamp in it that marks the job's start for the /tmp check at exit.
# Paths go to remote shells as arguments: Slurm sets TMPDIR=/tmp inside srun steps.
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    on_every_node bash -c 'mkdir -p "$1" && touch "$1/.job-start"' _ "$TMPDIR"
else
    mkdir -p "$TMPDIR" && touch "$TMPDIR/.job-start"
fi

# The first hardware thread of every physical core (0-39 on Kathleen).
_fmm_job_cores=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 |
    sort -n | paste -sd, -)
taskset -cp "$_fmm_job_cores" $$ >/dev/null
printf '=== batch shell bound to %s CPUs: %s\n' \
    "$(printf '%s\n' "$_fmm_job_cores" | tr ',' '\n' | wc -l)" "$(taskset -cp $$ | sed 's/.*: //')"

# step <name> <command...>: runs the command (through bash -c when it is one string with
# shell syntax), times it and records the result.
step() {
    local name=$1 start end status seconds
    shift
    printf '\n=== step %s: %s\n' "$name" "$*"
    start=$(date +%s%N)
    status=0
    if [ "$#" -eq 1 ]; then
        bash -c "$1" </dev/null || status=$?
    else
        "$@" </dev/null || status=$?
    fi
    end=$(date +%s%N)
    # Seconds with milliseconds.
    seconds=$(printf '%d.%03d' $(((end - start) / 1000000000)) $((((end - start) / 1000000) % 1000)))
    if [ "$status" -eq 0 ]; then
        printf '=== step %s: passed in %s s\n' "$name" "$seconds"
        _fmm_job_steps+="| $name | passed | $seconds |"$'\n'
    else
        printf '=== step %s: FAILED (exit %d) in %s s\n' "$name" "$status" "$seconds"
        _fmm_job_steps+="| $name | failed (exit $status) | $seconds |"$'\n'
        _fmm_job_failed=$((_fmm_job_failed + 1))
    fi
}

# On one node, with the job's TMPDIR as $1: its size; what this user created in /tmp
# since the stamp (listed, then removed); then the directory itself. Refuses any path
# that is not a per-job directory in /dev/shm.
_fmm_job_cleanup='
case "$1" in /dev/shm/fmm-*-[0-9]*) ;; *) echo "cleanup: refusing $1"; exit 1 ;; esac
printf "%s: TMPDIR %s\n" "$(hostname -s)" "$(du -sh "$1" 2>/dev/null | cut -f1)"
if [ -e "$1/.job-start" ]; then
    find /tmp -mindepth 1 -maxdepth 1 -user "$(id -u)" -newer "$1/.job-start" \
        -printf "%p %s bytes\n" -exec rm -rf {} + 2>/dev/null | sed "s|^|$(hostname -s): /tmp: |"
fi
rm -rf "$1"'

_fmm_job_end() {
    local status=$?
    printf '\n=== TMPDIR %s at the end, and what this job created in /tmp, per node:\n' "$TMPDIR"
    if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
        on_every_node bash -c "$_fmm_job_cleanup" _ "$TMPDIR"
    else
        bash -c "$_fmm_job_cleanup" _ "$TMPDIR"
    fi
    printf '\n=== steps of job %s\n| step | result | seconds |\n| --- | --- | ---: |\n%s' \
        "$SLURM_JOB_ID" "$_fmm_job_steps"
    printf '=== end %s, %d s in the job; %d step(s) failed\n' "$(date -Iseconds)" \
        $(($(date +%s) - _fmm_job_start)) "$_fmm_job_failed"
    if [ "$_fmm_job_failed" -gt 0 ] && [ "$status" -eq 0 ]; then
        status=1
    fi
    exit "$status"
}
trap _fmm_job_end EXIT
