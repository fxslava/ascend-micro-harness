#!/usr/bin/env bash
set -euo pipefail
root=/work/fused-three-zone-20261103/evidence
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
cmake -S "$root/src" -B "$root/build" -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2 -DSOC_VERSION=Ascend950PR_9599 >"$root/configure.log" 2>&1
cmake --build "$root/build" --target test_camodel_fused_decode_opt turboquant_fused_decode_opt_npu -j4 >"$root/build.log" 2>&1 || { tail -100 "$root/build.log"; exit 1; }
