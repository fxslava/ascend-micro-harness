#!/usr/bin/env bash
# Read the toolkit's OWN declarations for the Ascend C objects a change depends on.
#
#   bash scripts/probe_ascendc_headers.sh {bufpool|cube|tpipe-reset|all}
#
# Start with the API reference mirror, not here: C:\PipeSim\docs\ascendc_api (or
# $ASCENDC_API_DOCS) answers "what does this API do, on which products, under what
# constraints" without a toolkit at all. This script is for the cases where that is not
# enough, which in CANN 9.2 are specific and recurring:
#
#   * the reference and the shipped headers disagree on FIELD NAMES. MmadParams names the
#     C-init field cmatrixInitVal and not the reference's ccubeInitVal; Dn2NzParams names the
#     per-matrix strides srcDnMatrixStride/dstNzMatrixStride and not ...CubeStride. A field a
#     kernel relies on is read off the header, not off the doc.
#   * the reference documents an API's contract but not its IMPLEMENTATION, and sometimes the
#     implementation is the answer. Whether the Cube GEMM reduction may overlay the attention
#     phase's dead UB turns on what TPipe::Reset() and TPipe::InitBufPool()'s shared overload
#     actually do to startAddr_ -- which is in kernel_tpipe_impl.h and nowhere else.
#
# This consolidates three single-question probes written during the Cube GEMM reduction work
# (probe_bufpool_api.sh, probe_cube_api.sh, probe_tpipe_reset.sh), one topic each, so the
# questions and the exact greps that answered them stay reproducible in one place instead of
# in three throwaway files. It only reads headers: no build, no device, no simulator. Run it
# wherever the toolkit is -- the CANN 9.2 container or a WSL instance that has it.
set -uo pipefail

toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
inc="$toolkit/x86_64-linux/asc/include"
impl="$toolkit/x86_64-linux/asc/impl"
for dir in "$inc" "$impl"; do
    [[ -d "$dir" ]] || { printf 'missing toolkit directory: %s\n' "$dir" >&2; exit 2; }
done

# TBufPool: does it offer InitBuffer, and is TBufPool<VECCALC> the type to declare? This is
# what the reduction's overlay is built on -- it allocates its panels from a pool rather than
# from the TPipe, so the pool's own interface has to carry InitBuffer.
probe_bufpool() {
    printf '### every InitBuffer declaration in kernel_tpipe.h, with its enclosing context\n'
    grep -n 'InitBuffer\|^class \|^template \|EXTERN_IMPL_BUFPOOL\|^};' \
        "$inc/basic_api/kernel_tpipe.h" | sed -n '1,80p'
    printf '\n### the InitBufPool overloads: the shared one is the alias\n'
    grep -n 'InitBufPool' "$inc/basic_api/kernel_tpipe.h"
}

# The event-id pool, verbatim from probe_cube_api.sh. The question it answered was who owns
# which HardEvent id and where TPipe primes them at construction -- the reference documents
# neither, and the batched reduction's flag-id budget depends on both.
probe_cube() {
    printf '### FetchEventID / the event-id pool: who owns which id
'
    grep -rn --include='*.h' -B3 -A 30 'FetchEventID' "$impl/basic_api/kernel_tpipe"* 2>/dev/null | head -70
    printf '
### where TPipe primes events at construction (SetFlag in tpipe impl)
'
    grep -rn --include='*.h' -B6 -A 10 'SetFlag' "$impl/basic_api/kernel_tpipe"* 2>/dev/null | head -60
    printf '
### the event-id constants
'
    grep -rn --include='*.h' -e 'EVENT_ID[0-9]' -e 'QUE_MAX_EVENT' -e 'eventIdPool'         -e 'MAX_EVENT_ID' "$inc" "$impl" 2>/dev/null | head -20
}

# What TPipe::Reset() and TPipe::Destroy() actually do. This decided whether the batched
# reduction's UB can overlay the attention phase's dead buffers, which is the only way it fits
# the production operator's 4256-byte headroom: the answer was that it cannot be done by
# resetting the pipe, and the shared InitBufPool overload is what does it instead.
probe_tpipe_reset() {
    printf '### TPipe::Reset implementation\n'
    grep -n 'inline void TPipe::Reset' -A 30 "$impl/basic_api/kernel_tpipe_impl.h"
    printf '\n### TPipe::Destroy implementation\n'
    grep -n 'inline void TPipe::Destroy()' -A 14 "$impl/basic_api/kernel_tpipe_impl.h"
    printf '\n### InitBufPool: what the shared overload does to startAddr_\n'
    grep -n 'InitBufPool' -A 24 "$impl/basic_api/kernel_tpipe_impl.h" | sed -n '1,80p'
}

case "${1:-all}" in
    bufpool) probe_bufpool;;
    cube) probe_cube;;
    tpipe-reset) probe_tpipe_reset;;
    all)
        printf '========== bufpool ==========\n'; probe_bufpool
        printf '\n========== cube ==========\n'; probe_cube
        printf '\n========== tpipe-reset ==========\n'; probe_tpipe_reset
        ;;
    *) printf 'usage: %s {bufpool|cube|tpipe-reset|all}\n' "$0" >&2; exit 2;;
esac
