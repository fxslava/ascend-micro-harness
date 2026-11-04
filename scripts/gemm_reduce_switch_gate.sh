#!/usr/bin/env bash
# The switch gate: configuring with -DTQ_GEMM_REDUCE=ON must FAIL the operator's device build,
# and it must fail on the UB deficit rather than on anything else.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_switch_gate.sh
#
# This is a NEGATIVE control, and it is the gate that turns the deficit from a sentence in a
# report into a compile-time object. A clean build here would mean one of two things: the
# operator had silently gained 19648 bytes of UB headroom, or the static_assert had been
# removed. Either way the number in the report would be wrong and nothing else would notice.
#
# It builds through CMake rather than invoking ccec by hand, because the arch, the tikcfw
# include set and the host-stub forwarding all come from CANN's own ascendc_library macros --
# a hand-rolled compile fails on `Unsupported CCE architecture: dav-3510` and proves nothing.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
soc="${SOC_VERSION:-Ascend950PR_9599}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-/work/build-switch}"
log="$build.log"

rm -rf "$build"
cmake -S "$src" -B "$build" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSOC_VERSION="$soc" \
    -DTQ_GEMM_REDUCE=ON >"$log" 2>&1 || {
    printf 'FAIL switch gate: configure failed, so the switch was never compiled\n'
    tail -20 "$log"; exit 2; }

set +e
cmake --build "$build" --target camodel_fused_decode_opt_kernel -j8 >>"$log" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    printf 'FAIL switch gate: the Cube GEMM reduction path COMPILED with TQ_GEMM_REDUCE=ON.\n'
    printf '     Either the operator gained UB headroom (check kOperatorUbHeadroomBytes in\n'
    printf '     include/turboquant_gemm_reduce.hpp) or the deficit static_assert was removed.\n'
    exit 1
fi
if grep -q "does not fit this operator's UB headroom" "$log"; then
    printf 'PASS switch gate: TQ_GEMM_REDUCE=ON fails on the UB deficit, as designed\n'
    grep -m1 -B1 -A4 "does not fit this operator's UB headroom" "$log" | sed 's/^/     /'
    exit 0
fi
printf 'INCONCLUSIVE switch gate: the build failed, but not on the UB deficit assert, so the\n'
printf '     19648-byte number is unverified and something else is in the way.\n'
grep -m10 -E 'error:' "$log" | sed 's/^/     /'
exit 2
