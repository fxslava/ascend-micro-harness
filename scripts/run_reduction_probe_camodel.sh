#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
sim_lib="$toolkit/tools/simulator/$soc/lib"
msprof="$toolkit/tools/profiler/bin/msprof"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${REDUCTION_PROBE_OUT:-$repo/reports/reduction_probe_$stamp}"
fixed_trace="$repo/reports/reduction_probe_trace.json"

for f in "$toolkit/set_env.sh" "$sim_lib/libruntime_camodel.so" "$sim_lib/config.json" \
         "$sim_lib/config_stars.json" "$msprof"; do
    [[ -f "$f" ]] || { echo "missing CAModel component: $f" >&2; exit 2; }
done
[[ ! -e "$out" ]] || { echo "output already exists: $out" >&2; exit 2; }
[[ ! -e "$fixed_trace" ]] || { echo "refusing to overwrite $fixed_trace" >&2; exit 2; }
mkdir -p "$out"

build="${REDUCTION_PROBE_BUILD_DIR:-}"
if [[ -z "$build" ]]; then
    work="$(mktemp -d /tmp/tq-reduction-probe.XXXXXX)"
    mkdir -p "$work/src"
    rsync -a --exclude=.git --exclude=reports --exclude='build*' --exclude=bin \
        --exclude=__pycache__ --exclude=sim_log "$repo/" "$work/src/"
    cmake -S "$work/src" -B "$work/build" -G 'Unix Makefiles' \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="$soc" -DASCEND_HOME_PATH="$toolkit" \
        >"$out/configure.log" 2>&1
    cmake --build "$work/build" --target test_camodel_reduction_probe -j "${BUILD_JOBS:-4}" \
        >"$out/build.log" 2>&1
    build="$work/build"
else
    [[ -x "$build/lib/test_camodel_reduction_probe" ]] || {
        echo "missing executable: $build/lib/test_camodel_reduction_probe" >&2; exit 2; }
    work="$(mktemp -d /tmp/tq-reduction-probe-run.XXXXXX)"
fi

mkdir -p "$work/run" "$work/profile" "$work/config"
cp "$sim_lib/config.json" "$work/config/"
cp "$sim_lib/config_stars.json" "$work/config/"
cat >"$work/run/application.sh" <<APP
#!/usr/bin/env bash
set -euo pipefail
exec "$build/lib/test_camodel_reduction_probe"
APP
chmod 700 "$work/run/application.sh"

unset LD_PRELOAD LD_LIBRARY_PATH CAMODEL_CONFIG_PATH
set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"
export LD_LIBRARY_PATH="$build/lib:$sim_lib:$toolkit/lib64:$toolkit/devlib"
export CAMODEL_CONFIG_PATH="$work/config"
cp "$work/run/application.sh" "$out/application.sh"
cd "$work/run"
"$msprof" op simulator "--application=$work/run/application.sh" \
    "--output=$work/profile" "--soc-version=$soc" \
    --kernel-name=turboquant_fused_decode_opt_reduction_probe --launch-count=1 \
    --timeout="${PROFILE_TIMEOUT_MINUTES:-30}" >"$out/msprof.log" 2>&1
grep -Fq 'max_ulp=0' "$out/msprof.log"
grep -Fq 'neutral_partials=448' "$out/msprof.log"
cp "$out/msprof.log" "$out/probe.log"
mapfile -d '' traces < <(find "$work/profile" -path '*/simulator/trace.json' -type f -print0)
[[ ${#traces[@]} -eq 1 ]] || { echo "expected one CAModel trace, found ${#traces[@]}" >&2; exit 1; }
python3 "$repo/scripts/validate_camodel_trace.py" "${traces[0]}" \
    --summary "$out/trace-validation.json"
cp "${traces[0]}" "$out/reduction_probe_trace.json"
cp "${traces[0]}" "$fixed_trace"
printf '%s\n' "$soc" >"$out/soc.txt"
printf '%s\n' "$build" >"$out/build-dir.txt"
printf 'PASS: synthetic Split-K reduction CAModel probe and trace export\n'
printf '  Probe result: %s\n' "$out/probe.log"
printf '  Perfetto trace: %s\n' "$fixed_trace"
printf '  Trace validation: %s\n' "$out/trace-validation.json"
