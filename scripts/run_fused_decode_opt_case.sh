#!/usr/bin/env bash
# Additional fixture cases against a previously built kernel and host runner.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(realpath "$1")"
tag="$2"
shift 2
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]]
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
set +u
source "$toolkit/set_env.sh"
set -u
sim="$toolkit/tools/simulator/${SOC_VERSION:-Ascend950PR_9599}/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
runner="${FUSED_DECODE_OPT_RUNNER:-$root/build/lib/test_camodel_fused_decode_opt_updated}"
# RESUMABLE, because the full ladder is hours of model time and a dropped shell used to
# cost all of it. A case whose directory already holds its PASS line is skipped; a case
# whose directory exists WITHOUT one is an interrupted launch, so it is discarded and
# redone. FUSED_DECODE_OPT_FORCE=1 re-runs every case regardless.
#
# This is what turns a second run of scripts/run_fused_decode_canary_ladder.sh against
# the same FUSED_DECODE_OPT_ROOT into a resume rather than a restart.
if [[ -z "${FUSED_DECODE_OPT_FORCE:-}" && -f "$root/$tag/run.log" ]] &&
        grep -q '^PASS fused_decode_opt' "$root/$tag/run.log"; then
    printf 'SKIP %s: already passed, see %s/run.log\n' "$tag" "$root/$tag"
    exit 0
fi
rm -rf "$root/$tag"
mkdir -p "$root/$tag"
printf '%q ' "$runner" "$@" >"$root/$tag/command.txt"
printf '\n' >>"$root/$tag/command.txt"
status=0
(cd "$root/$tag"; timeout 1800s "$runner" "$@" >run.log 2>&1) || status=$?
cat "$root/$tag/run.log"
if ((status)); then exit "$status"; fi
if grep -Ei '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/$tag/run.log"; then exit 1; fi
python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" "$root/$tag" >"$root/$tag/audit.log"
printf 'PASS trace audit: %s/%s/audit.log\n' "$root" "$tag"
