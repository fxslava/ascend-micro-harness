#!/usr/bin/env bash
set -euo pipefail
cd "/tmp/ascend-fused-opt-json.EQO5lZ/run"
exec "/home/dmin/work/three-zone-check-20261103/build/lib/bench_decode_splitk_sim" --impl "fused-opt" --single --context "128" --page-tokens 128 --heads 1 --rows "16" --requests 1 --splitk 1 --csv-out "/mnt/c/Ascend/ascend-micro-harness/reports/decode_comparison_threezone_128_20261103T233843Z/results.csv"
