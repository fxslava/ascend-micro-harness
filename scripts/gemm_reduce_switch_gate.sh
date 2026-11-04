#!/usr/bin/env bash
# The switch gate: with the overlay in place, -DTQ_GEMM_REDUCE=ON must BUILD. Before the overlay
# it had to fail, on a static_assert naming the operator's 19648-byte UB deficit; the deficit is
# still computed and still true, and what changed is that the reducer no longer spends leftover
# UB -- it allocates from a TBufPool aliased onto the attention phase's transient pool.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_switch_gate.sh
#
# Two things are checked, because a build that merely succeeds does not say the overlay is the
# reason it succeeded:
#
#   1. TQ_GEMM_REDUCE=ON builds clean at -Wall -Werror.
#   2. The NEGATIVE control: with the pool's length cut below what the reducer needs, the build
#      must fail on the reducer's own "must fit inside it" assert. That is what proves the fit
#      is checked rather than assumed -- if the reducer ever outgrows the pool, this is the
#      error that says so instead of an overlap nobody notices.
#
# It builds through CMake rather than invoking ccec by hand, because the arch, the tikcfw include
# set and the host-stub forwarding all come from CANN's own ascendc_library macros -- a
# hand-rolled compile fails on `Unsupported CCE architecture: dav-3510` and proves nothing.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
soc="${SOC_VERSION:-Ascend950PR_9599}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-/work/build-switch}"
log="$build.log"
status=0

printf '=== 1. TQ_GEMM_REDUCE=ON must build ===\n'
rm -rf "$build"
cmake -S "$src" -B "$build" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSOC_VERSION="$soc" \
    -DTQ_GEMM_REDUCE=ON >"$log" 2>&1 || {
    printf 'FAIL switch gate: configure failed\n'; tail -20 "$log"; exit 2; }
if cmake --build "$build" --target camodel_fused_decode_opt_kernel -j8 >>"$log" 2>&1; then
    printf 'PASS the operator builds with the Cube GEMM reduction driving its reduction phase\n'
else
    printf 'FAIL switch gate: TQ_GEMM_REDUCE=ON does not build\n'
    grep -m10 -E 'error' "$log" | sed 's/^/     /'
    status=1
fi

printf '\n=== 2. negative control: a pool too short must be rejected ===\n'
neg="$build-neg"
rm -rf "$neg" "$neg.src"
cp -r "$src" "$neg.src"
# Halve the restated pool length. The reducer needs 27264 bytes and would then be offered
# 16384, so its own assert is the one that must fire.
sed -i 's|^constexpr std::uint32_t kGemmSharedPoolBytes = .*|constexpr std::uint32_t kGemmSharedPoolBytes = 16384;|' \
    "$neg.src/include/turboquant_gemm_reduce.hpp"
# The restatement pin would fire first and hide the fit check, so relax it for this control
# only -- the point is to reach the reducer's assert, not to test the pin twice.
sed -i 's|^static_assert(kGemmSharedPoolBytes == 32768.*|static_assert(kGemmSharedPoolBytes == 16384, "negative control");|' \
    "$neg.src/include/turboquant_gemm_reduce.hpp"
sed -i 's|ascend_micro::fused_decode_opt::kTransientPoolBytes ==|16384u ==|' \
    "$neg.src/kernels/ops/turboquant/turboquant_fused_decode_opt.cpp"
cmake -S "$neg.src" -B "$neg" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSOC_VERSION="$soc" \
    -DTQ_GEMM_REDUCE=ON >"$neg.log" 2>&1 || {
    printf 'INCONCLUSIVE: the control did not configure\n'; tail -10 "$neg.log"; exit 2; }
if cmake --build "$neg" --target camodel_fused_decode_opt_kernel -j8 >>"$neg.log" 2>&1; then
    printf 'FAIL negative control: a 16384-byte pool was accepted for a 27264-byte reducer.\n'
    printf '     The fit is not being checked.\n'
    status=1
elif grep -q 'must fit inside it' "$neg.log"; then
    printf 'PASS negative control: a pool shorter than the reducer is rejected by its own assert\n'
    grep -m1 -o 'static assertion failed.*must fit inside it' "$neg.log" | cut -c1-150 | sed 's/^/     /'
else
    printf 'INCONCLUSIVE negative control: it failed, but not on the fit assert\n'
    grep -m6 -E 'error' "$neg.log" | sed 's/^/     /'
    status=2
fi
rm -rf "$neg" "$neg.src"
exit "$status"
