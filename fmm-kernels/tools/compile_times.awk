# Sums the first launch of each kernel variant in a CubeCL profiling log
# (CUBECL_DEBUG_LOG): that launch includes the variant's compilation.
BEGIN { FS = "|" }
NF >= 4 && $2 ~ /[0-9](ns|µs|ms|s) *$/ {
    t = $2; gsub(/ /, "", t); key = $3 "|" $4
    if (key in seen) next
    seen[key] = 1
    v = t + 0
    if (t ~ /ms$/) v /= 1e3; else if (t ~ /µs$/) v /= 1e6; else if (t ~ /ns$/) v /= 1e9
    total += v; n++
    if (v > max) { max = v; slowest = key }
}
END { printf "kernel variants: %d; first launches (compilation included): %.2f s; slowest %.3f s (%s)\n", n, total, max, slowest }
