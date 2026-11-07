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
mkdir -p "$root/cli" "$root/csv-compat"
"$bench" --help >"$root/cli/help.txt" 2>&1
grep -q 'fused-opt' "$root/cli/help.txt"
if "$bench" --impl fused-opt --splitk 12 >"$root/cli/invalid-splits.txt" 2>&1; then exit 1; fi
grep -q 'splitk must be 1..8' "$root/cli/invalid-splits.txt"
# --impl all used to reach the page-tokens guard here because the simulator benchmark was
# built with the legacy decoders in. It is the fused operator alone now, so `all` is refused
# earlier, by the isolation guard; --impl fused-opt is what reaches the page-tokens rule.
if "$bench" --impl all >"$root/cli/invalid-impl.txt" 2>&1; then exit 1; fi
grep -q 'benchmarks the fused decode operator alone' "$root/cli/invalid-impl.txt"
if "$bench" --impl fused-opt --page-tokens 16 >"$root/cli/invalid-page.txt" 2>&1; then exit 1; fi
grep -q 'unsupported page-tokens' "$root/cli/invalid-page.txt"
if "$bench" --impl fused-opt --rows 17 >"$root/cli/invalid-rows.txt" 2>&1; then exit 1; fi
grep -q 'rows 1..16' "$root/cli/invalid-rows.txt"
if [[ "${2:-}" == cli-only ]]; then
    printf 'PASS help and fused-opt split/page/row CLI guards\n'
    exit 0
fi
# The CSV compatibility gate. It used to run --impl aiv and assert that the opt_* columns
# came back NA; the AIV decoder is deleted, so it runs the one implementation this binary
# has and asserts the other direction -- that the aiv_* columns are still PRESENT and NA,
# which is what lets a reader line a run in reports/ up with a run taken today.
(cd "$root/csv-compat"; timeout 1800s "$bench" --impl fused-opt --single --context 64 \
    --page-tokens 64 --heads 1 --rows 16 --splitk 1 --csv-out results.csv >run.log 2>&1)
grep -q 'AIC=0 OPT=1' "$root/csv-compat/run.log"
if grep -Ei '\[error\]|\[aicore error\]|mismatch|FAIL' "$root/csv-compat/run.log"; then exit 1; fi
python3 - "$root/csv-compat/results.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as source:
    reader = csv.DictReader(source)
    columns, rows = reader.fieldnames, list(reader)
for name in ('aiv_us', 'aiv_speedup', 'aiv_kv_gbs', 'aiv_launches'):
    assert name in columns, name
assert len(rows) == 1 and rows[0]['opt_launches'] == '1'
assert rows[0]['aiv_us'] == rows[0]['aiv_speedup'] == rows[0]['aiv_kv_gbs'] == 'NA'
assert rows[0]['aic_batch_us'] == 'NA'
print('PASS CLI guards, help, fused-opt dispatch/reference and CSV compatibility')
PY
