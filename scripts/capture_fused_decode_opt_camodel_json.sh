#!/usr/bin/env bash
# Native CANN/msprof Chrome-trace JSON for a full-width decode launch.
# Reuses a completed build from build_fused_decode_opt_bench.sh.
# Optional arguments: <build-root> <fused-opt|aic> <context>, defaults fused-opt/384.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$(realpath "${1:?pass the benchmark build root}")"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
mode="${2:-fused-opt}"
context="${3:-384}"
case "$mode" in
    fused-opt) kernel=turboquant_mm_fused_decode_opt_kv4fp8_half; rows=16; expected='AIV=0 AIC=0 OPT=1'; library=libcamodel_fused_decode_opt_kernel.so;;
    aic) kernel=turboquant_paged_decode_splitk_aic; rows=1; expected='AIV=0 AIC=1 OPT=0'; library=libcamodel_paged_decode_attention_kernel.so;;
    *) printf 'Expected fused-opt or aic\n' >&2; exit 2;;
esac
[[ "$context" =~ ^[0-9]+$ && "$context" -ge 128 && "$((context % 128))" == 0 ]]
work="$(mktemp -d /tmp/ascend-fused-opt-json.XXXXXX)"
out="${FUSED_OPT_TRACE_OUT:-$repo/reports/decode_comparison_${mode}_${context}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$out" ]] || { printf 'Output already exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$out" "$work/run" "$work/config"
out="$(realpath "$out")"
printf 'Trace work root: %s\nTrace output: %s\n' "$work" "$out"
set +u
source "$toolkit/set_env.sh"
set -u
sim="$toolkit/tools/simulator/$soc/lib"
bench="$build_root/build/lib/bench_decode_splitk_sim"
[[ -x "$bench" ]]
export LD_LIBRARY_PATH="$build_root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
cp "$sim/config.json" "$sim/config_stars.json" "$work/config/"
export CAMODEL_CONFIG_PATH="$work/config"
cp -a "$work/config" "$out/config"
{
    printf 'git_commit='; git -C "$repo" rev-parse HEAD
    printf 'build_root=%s\nwork=%s\nsoc=%s\nkernel=%s\n' "$build_root" "$work" "$soc" "$kernel"
    printf 'impl=%s\ncontext=%s\npage_tokens=128\nkv_heads=1\nshared_query_rows=16\nsplits=1\nrequests=1\n' "$mode" "$context"
    printf 'LD_LIBRARY_PATH=%s\nCAMODEL_CONFIG_PATH=%s\n' "$LD_LIBRARY_PATH" "$CAMODEL_CONFIG_PATH"
    sha256sum "$bench" "$build_root/build/lib/$library"
} >"$out/environment.txt"
cat >"$work/run/application.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$work/run"
exec "$bench" --impl "$mode" --single --context "$context" --page-tokens 128 --heads 1 --rows "$rows" --requests 1 --splitk 1 --csv-out "$out/results.csv"
EOF
chmod +x "$work/run/application.sh"
cp "$work/run/application.sh" "$out/application.sh"
command=("$toolkit/tools/profiler/bin/msprof" op simulator
    "--application=$work/run/application.sh" "--output=$work/profile"
    "--soc-version=$soc" --aic-metrics=PipeUtilization
    "--kernel-name=$kernel" --launch-count=1 --timeout=10)
printf '%q ' "${command[@]}" >"$out/command.txt"
printf '\n' >>"$out/command.txt"
status=0
(cd "$work/run"; "${command[@]}" >"$out/profile.log" 2>&1) || status=$?
printf '%s\n' "$status" >"$out/profile-exit-code.txt"
if ((status)); then tail -80 "$out/profile.log"; exit "$status"; fi
grep -q "$expected" "$out/profile.log"
if grep -Ei '\[error\]|\[aicore error\]|mismatch|FAIL:' "$out/profile.log"; then exit 1; fi
mapfile -t traces < <(find "$work/profile" -path '*/simulator/trace.json' -type f)
[[ "${#traces[@]}" == 1 ]] || { printf 'Expected one native trace, found %s\n' "${#traces[@]}" >&2; exit 1; }
printf '%s\n' "${traces[0]}" >"$out/native-trace-path.txt"
cp "${traces[0]}" "$out/trace.json"
sha256sum "$out/trace.json" >"$out/trace.sha256"
python3 "$repo/scripts/validate_fused_decode_opt_json_trace.py" "$out/trace.json" >"$out/trace-validation.json"
printf 'PASS native CAModel timeline: %s/trace.json\n' "$out"
