#!/usr/bin/env bash
# Does TBufPool offer InitBuffer, and is TBufPool<VECCALC> the type to declare?
set -uo pipefail
inc=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/asc/include

printf '### every InitBuffer declaration in kernel_tpipe.h, with its enclosing context\n'
grep -n 'InitBuffer\|^class \|^template \|EXTERN_IMPL_BUFPOOL\|^};' "$inc/basic_api/kernel_tpipe.h" |
    sed -n '1,80p'
