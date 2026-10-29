#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline="${1:?usage: run_decode_pipeline_regressions.sh BASELINE_DIR BENCH_BUILD_DIR}"
bench_build="${2:?provide the clean bench_decode_splitk_sim build directory}"
mode="${3:-all}"
[[ "$mode" == all || "$mode" == workers-only ]]
[[ -x "$baseline/build/lib/test_camodel_single_token_decode_aiv" ]]
[[ -x "$bench_build/lib/bench_decode_splitk_sim" ]]
# Explicitly require all ten isolated/depth/wrap passes before sequence tests.
[[ $(grep -c '^PASS ' "$baseline/summary.txt") == 10 ]]
set +u
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
set -u
export LD_LIBRARY_PATH="$baseline/build/lib:$bench_build/lib:/usr/local/Ascend/cann-9.2.0-beta.2/tools/simulator/Ascend950PR_9599/lib:/usr/local/Ascend/cann-9.2.0-beta.2/lib64:/usr/local/Ascend/cann-9.2.0-beta.2/devlib:${LD_LIBRARY_PATH:-}"
if [[ "$mode" == all ]]; then
for case in rows6 pages2 odd3; do
  dir="$baseline/regression_$case"
  mkdir -p "$dir"
  case "$case" in
    rows6) args=(--rows 6 --pages 1 --strips 1); rows=6; strips=1;;
    pages2) args=(--rows 1 --pages 2 --strips 2); rows=1; strips=4;;
    odd3) args=(--rows 1 --pages 1 --strips 3); rows=1; strips=3;;
  esac
  (
    cd "$dir"
    timeout 1800 "$baseline/build/lib/test_camodel_single_token_decode_aiv" --head 2 --no-fixture-check "${args[@]}" > run.log 2>&1
    ! grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' run.log
    shopt -s nullglob
    for dump in core*.excp_log.dump excp_log*.dump; do [[ ! -s "$dump" ]]; done
    python3 "$repo/scripts/analyze_decode_ring_trace.py" . --strips "$strips" > audit.json
    python3 "$repo/scripts/analyze_single_token_decode_trace.py" core0.veccore0.instr_popped_log.dump core0.veccore0.instr_log.dump --rows "$rows" --strips "$strips" --items 1 > census.txt
  )
  echo "PASS regression $case"
done
fi
for k in 2 4; do
  dir="$baseline/regression_splitk$k"
  mkdir -p "$dir"
  (
    cd "$dir"
    # Five strips per worker at K=2 crosses both ring slots with an odd tail.
    # K=4 over three pages also includes an empty worker.
    tokens=160
    if [[ $k == 4 ]]; then tokens=48; fi
    timeout 1800 "$bench_build/lib/bench_decode_splitk_sim" --context "$tokens" --page-tokens 16 --heads 1 --rows 1 --splitk "$k" --single --impl aiv > run.log 2>&1
    ! grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' run.log
    shopt -s nullglob
    for dump in core*.excp_log.dump excp_log*.dump; do [[ ! -s "$dump" ]]; done
    python3 "$repo/scripts/analyze_decode_ring_trace.py" . > audit.json
    python3 "$repo/scripts/analyze_splitk_decode_trace.py" . --assert-batched > atomic.json
  )
  echo "PASS regression splitk$k"
done
