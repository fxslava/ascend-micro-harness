#!/usr/bin/env bash
set -euo pipefail
cd "/tmp/ascend-fused-opt-json.GtN1UB/run"
exec "/tmp/ascend-fused-opt-bench.yOuivK/build/lib/bench_decode_splitk_sim" --impl "fused-opt" --single --context "768" --page-tokens 128 --heads 1 --rows "16" --requests 1 --splitk 1 --csv-out "/mnt/c/Ascend/ascend-micro-harness/reports/decode_comparison_fused-opt_768_20261103T182824Z/results.csv"
