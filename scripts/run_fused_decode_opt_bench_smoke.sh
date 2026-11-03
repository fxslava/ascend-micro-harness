#!/usr/bin/env bash
# One real benchmark dispatch: two requests, two KV heads, odd GQA group=3,
# three fragmented pages and K=2 (eight physical MIX blocks).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(realpath "$1")"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
set +u
source "$toolkit/set_env.sh"
set -u
sim="$toolkit/tools/simulator/${SOC_VERSION:-Ascend950PR_9599}/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
mkdir -p "$root/fused-opt-smoke"
cd "$root/fused-opt-smoke"
status=0
timeout 1800s "$root/build/lib/bench_decode_splitk_sim" --impl fused-opt --single \
    --context 192 --page-tokens 64 --requests 2 --heads 2 --rows 3 --splitk 2 \
    --csv-out results.csv >run.log 2>&1 || status=$?
cat run.log
if ((status)); then exit "$status"; fi
if grep -Ei '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' run.log; then exit 1; fi
grep -q 'AIV=0 AIC=0 OPT=1' run.log
python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" "$PWD" >audit.json
python3 - <<'PY'
import csv
with open('results.csv') as source:
    rows = list(csv.DictReader(source))
assert len(rows) == 1
r = rows[0]
assert r['opt_grid'] == '8' and r['opt_launches'] == '1'
assert r['aiv_launches'] == r['aic_launches'] == '0'
assert r['opt_us'] == r['opt_speedup'] == r['opt_kv_cache_gbs'] == 'NA'
print('PASS fused-opt benchmark dispatch, CPU reference, guards, eight-block trace audit, CSV and unavailable simulator timings')
PY
