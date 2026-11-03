#!/usr/bin/env bash
set -euo pipefail
cd "/tmp/ascend-fused-opt-json.FI4jQn/run"
exec "/tmp/ascend-fused-opt-bench.yOuivK/build/lib/bench_decode_splitk_sim" --impl "aic" --single --context "128" --page-tokens 128 --heads 1 --rows "1" --requests 1 --splitk 1 --csv-out "/mnt/c/Ascend/ascend-micro-harness/reports/decode_comparison_aic_128_20261103T182054Z/results.csv"
