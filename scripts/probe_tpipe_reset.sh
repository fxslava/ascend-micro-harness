#!/usr/bin/env bash
# What TPipe::Reset() actually does. This decides whether the batched reduction's UB can
# overlay the attention phase's dead buffers, which is the only way it fits the production
# operator's 4256-byte headroom.
set -uo pipefail
impl=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/asc/impl

printf '### TPipe::Reset implementation\n'
grep -n 'inline void TPipe::Reset' -A 30 "$impl/basic_api/kernel_tpipe_impl.h"

printf '\n### TPipe::Destroy implementation\n'
grep -n 'inline void TPipe::Destroy()' -A 14 "$impl/basic_api/kernel_tpipe_impl.h"
