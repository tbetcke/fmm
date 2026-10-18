#!/bin/sh
# Runs K copies of the `level` example at once (one per core: the node's processes, each
# with its own tables and data, as T1's level benchmark did) and prints, per variant, the
# median over the copies of their medians, and the slowest copy.
#
#   tools/run_node.sh K <level example arguments>
#
# Build first: cargo build --release -p nd-fmm-spike-plane-wave --example level
# (from the workspace root; the binary is target/release/examples/level).
set -eu
k=$1
shift
root=$(cd "$(dirname "$0")/../../.." && pwd)
bin=${LEVEL_BIN:-$root/target/release/examples/level}
out=$(mktemp -d "${TMPDIR:-/tmp}/pw-node.XXXXXX")
i=0
while [ "$i" -lt "$k" ]; do
    "$bin" "$@" >"$out/$i.txt" &
    i=$((i + 1))
done
wait
cat "$out"/*.txt | awk -v k="$k" '
    {
        v = $1
        for (f = 2; f <= NF; f++) {
            split($f, kv, "=")
            if (kv[1] == "us_per_pair") t[v] = t[v] " " kv[2]
            if (kv[1] == "outgoing_ms") o[v] = o[v] " " kv[2]
            if (kv[1] == "incoming_ms") g[v] = g[v] " " kv[2]
            if (kv[1] == "p") p = kv[2]
            if (kv[1] == "s") s = kv[2]
            if (kv[1] == "level") l = kv[2]
        }
        if (!(v in seen)) { seen[v] = 1; order[++nv] = v }
    }
    function med(str,   a, n, i, j, x) {
        n = split(str, a, " ")
        for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (a[j] + 0 < a[i] + 0) { x = a[i]; a[i] = a[j]; a[j] = x }
        return a[int((n + 1) / 2)]
    }
    function mx(str,   a, n, i, m) {
        n = split(str, a, " "); m = 0
        for (i = 1; i <= n; i++) if (a[i] + 0 > m) m = a[i] + 0
        return m
    }
    END {
        for (i = 1; i <= nv; i++) {
            v = order[i]
            line = sprintf("%s p=%s s=%s level=%s copies=%d us_per_pair_median=%s us_per_pair_max=%.4f", v, p, s, l, k, med(t[v]), mx(t[v]))
            if (v in o) line = line sprintf(" outgoing_ms_median=%s incoming_ms_median=%s", med(o[v]), med(g[v]))
            print line
        }
    }'
rm -r "$out"
