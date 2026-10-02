# Inlining check of nd-fmm-simd (fmm-simd/CLAUDE.md, "Inlining check"): prints every
# loop of one function in a disassembly, that is every backward branch and the
# instructions from its target to it, with the instruction count and the number of
# calls (bl, blr, call) in the loop. A vector-layer method that is not inlined shows
# up as a call inside the loop.
#
# Usage, with the demangled function name in `f`:
#
#     objdump -d -C --no-show-raw-insn <release binary> \
#         | awk -v f='nd_fmm_simd::arch::neon::rsqrt_slice_f32' -f fmm-simd/tools/inner_loops.awk
#
# Works with llvm-objdump (macOS `objdump`) and GNU objdump. Addresses are compared as
# hex strings of equal length, which holds within one function.

/^[0-9a-f]+ <.*>:$/ { on = index($0, "<" f ">:") > 0; n = 0; next }

on && /^ *[0-9a-f]+:/ {
    a = $1
    sub(":", "", a)
    addr[++n] = a
    line[n] = $0
    if (match($0, /[0-9a-f]+ <[^>]*>$/)) {
        t = substr($0, RSTART, RLENGTH)
        sub(/ .*/, "", t)
        sub(/^0x/, "", t)
        if (length(t) == length(a) && t < a) {
            for (i = 1; i <= n && addr[i] != t; i++);
            calls = 0
            for (j = i; j <= n; j++) if (line[j] ~ /\t(bl|blr|call[a-z]*)\t/) calls++
            print f ": loop of " n - i + 1 " instructions, " calls " calls"
            for (j = i; j <= n; j++) print line[j]
        }
    }
}
