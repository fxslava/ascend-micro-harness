#!/usr/bin/env bash
# Read the toolkit's own declarations for the API objects the batched GEMM reduction depends on.
# Run inside the CANN 9.2 container.
#
# This exists because the API reference and the shipped headers disagree in this toolkit often
# enough to matter -- MmadParams names the C-init field cmatrixInitVal and not the doc's
# ccubeInitVal, and Dn2NzParams names the per-matrix strides srcDnMatrixStride/dstNzMatrixStride
# and not ...CubeStride. A field this prototype relies on is read off the header, not the doc.
set -uo pipefail
inc=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/asc/include
impl=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/asc/impl

printf '### FetchEventID / the event-id pool: who owns which id\n'
grep -rn --include='*.h' -B3 -A 30 'FetchEventID' "$impl/basic_api/kernel_tpipe"* 2>/dev/null | head -70

printf '\n### where TPipe primes events at construction (SetFlag in tpipe impl)\n'
grep -rn --include='*.h' -B6 -A 10 'SetFlag' "$impl/basic_api/kernel_tpipe"* 2>/dev/null | head -60

printf '\n### the event-id constants\n'
grep -rn --include='*.h' -e 'EVENT_ID[0-9]' -e 'QUE_MAX_EVENT' -e 'eventIdPool' \
    -e 'MAX_EVENT_ID' "$inc" "$impl" 2>/dev/null | head -20
