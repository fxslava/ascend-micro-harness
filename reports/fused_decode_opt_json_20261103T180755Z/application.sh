#!/usr/bin/env bash
set -euo pipefail
cd "/tmp/ascend-fused-opt-json.B2EzAV/run"
exec "/tmp/ascend-fused-opt-bench.yOuivK/build/lib/bench_decode_splitk_sim" --impl fused-opt --single --context 384 --page-tokens 128 --heads 1 --rows 16 --requests 1 --splitk 1 --csv-out "/mnt/c/Ascend/ascend-micro-harness/reports/fused_decode_opt_json_20261103T180755Z/results.csv"
