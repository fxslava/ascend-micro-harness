#!/usr/bin/env bash
set -euo pipefail
root="$(realpath "$1")"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
set +u
source "$toolkit/set_env.sh"
set -u
sim="$toolkit/tools/simulator/${SOC_VERSION:-Ascend950PR_9599}/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
bench="$root/build/lib/bench_decode_splitk_sim"
mkdir -p "$root/cli" "$root/legacy-aiv"
"$bench" --help >"$root/cli/help.txt" 2>&1
grep -q 'fused-opt' "$root/cli/help.txt"
if "$bench" --impl fused-opt --splitk 12 >"$root/cli/invalid-splits.txt" 2>&1; then exit 1; fi
grep -q 'splitk must be 1..8' "$root/cli/invalid-splits.txt"
if "$bench" --impl all --page-tokens 16 >"$root/cli/invalid-page.txt" 2>&1; then exit 1; fi
grep -q 'unsupported page-tokens' "$root/cli/invalid-page.txt"
if "$bench" --impl fused-opt --rows 17 >"$root/cli/invalid-rows.txt" 2>&1; then exit 1; fi
grep -q 'rows 1..16' "$root/cli/invalid-rows.txt"
if [[ "${2:-}" == cli-only ]]; then
    printf 'PASS help and fused-opt split/page/row CLI guards\n'
    exit 0
fi
(cd "$root/legacy-aiv"; timeout 1800s "$bench" --impl aiv --single --context 16 \
    --page-tokens 16 --heads 1 --rows 1 --splitk 1 --csv-out results.csv >run.log 2>&1)
grep -q 'AIV=1 AIC=0 OPT=0' "$root/legacy-aiv/run.log"
if grep -Ei '\[error\]|\[aicore error\]|mismatch|FAIL' "$root/legacy-aiv/run.log"; then exit 1; fi
python3 - "$root/legacy-aiv/results.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as source: rows = list(csv.DictReader(source))
assert len(rows) == 1 and rows[0]['opt_launches'] == '0'
assert rows[0]['opt_grid'] == rows[0]['opt_us'] == 'NA'
print('PASS CLI guards, help, legacy AIV dispatch/reference and CSV compatibility')
PY
