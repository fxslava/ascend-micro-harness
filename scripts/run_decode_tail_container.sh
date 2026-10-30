#!/usr/bin/env bash
# Build the unaligned single-token decode target for dav-3510 and run the
# single-tile canary rungs, inside the CANN 9.2 container.
#
# This is the only leg that can RUN the tail mask: the host gate compiles and
# checks arithmetic, but the masking itself is device code. Everything here
# needs the container, because the WSL instance's own CANN is 8.5.0 and knows
# neither dav-3510 nor Ascend950PR_9599.
#
#   bash scripts/run_decode_tail_container.sh            # build and run
#   bash scripts/run_decode_tail_container.sh <target>   # a different target
#
# From the repository root, with docker available:
#
#   docker run --rm -v "$(pwd)":/workspace -w /workspace \
#       ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 \
#       bash scripts/run_decode_tail_container.sh
#
# Why only S = 1 and S = 17
# -------------------------
# Those are the two shapes the tail mask has to get right: one live token among
# fifteen canary lanes, and a full strip followed by a strip holding a single
# live token. Longer contexts add pages, not tail cases, and a full-context run
# on CAModel costs hours -- the S=1/S=17 pair takes about nine minutes of model
# time. The complete ladder (S up to 1023) exists in the same harness without
# --single-tile-canary, for a machine with real silicon.
#
# What a pass means, and what to read besides the exit status
# ----------------------------------------------------------
# The runner's own PASS line covers the emitted context: finite, within
# tolerance of the FP64 oracle, and with every slot past seq_len still holding
# its 0x7FC0BEEF canary. But ALSO grep the log for `vec_err_idata_inf_nan`:
# the model reports NaN reaching a vector instruction's INPUT, which a kernel
# can do while still producing a clean answer if it overwrites the result. A
# passing run with those errors in it is not a clean run.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

target="${1:-test_decode_single_token_unaligned_sim}"

echo "=== configure ==="
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -6
echo "configure status: ${PIPESTATUS[0]}"

echo
echo "=== build ${target} ==="
cmake --build build -j8 --target "${target}" 2>&1 \
    | grep -E "error:|Built target|Linking CXX executable" | tail -20
echo "build status: ${PIPESTATUS[0]}"

echo
echo "=== run: single-tile canary ==="
if [ ! -x "build/lib/${target}" ]; then
    echo "SKIPPED: build/lib/${target} was not produced"
    exit 1
fi
timeout 3600 "./build/lib/${target}" --single-tile-canary
status=$?
echo "run status: ${status}"
exit "${status}"
