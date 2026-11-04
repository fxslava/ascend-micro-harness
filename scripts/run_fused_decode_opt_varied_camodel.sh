#!/usr/bin/env bash
# Separate regression: varying tile maxima exercise alpha < 1 and exp/merge.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(realpath "$1")"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
set +u
source "$toolkit/set_env.sh"
set -u
bash "$repo/scripts/build_fused_decode_opt_runner.sh" "$root"
sim="$toolkit/tools/simulator/${SOC_VERSION:-Ascend950PR_9599}/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
mkdir -p "$root/varied-7"
status=0
(cd "$root/varied-7"; timeout 1800s "$root/build/lib/test_camodel_fused_decode_opt_updated" \
    --tiles 7 --varied 1 >run.log 2>&1) || status=$?
cat "$root/varied-7/run.log"
if ((status)); then exit "$status"; fi
if grep -Ei '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/varied-7/run.log"; then exit 1; fi
python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" "$root/varied-7" >"$root/varied-7/audit.log"
printf 'PASS varied-score trace audit: %s/varied-7/audit.log\n' "$root"
