#!/usr/bin/env bash
# Native CANN export: no repository parser of instr_popped_log.dump.
# Usage: bash scripts/profile_camodel_trace.sh [new-output-directory]
# Defaults: one AIC worker, one supported 16-token diagnostic page/strip,
# M=16, d=256, one launch. Set PAGE_TOKENS and CONTEXT to 128 for a full page.
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, CAMODEL_CONFIG_PATH (directory),
# CAMODEL_TRACE_CONTEXT, CAMODEL_TRACE_PAGE_TOKENS, CAMODEL_TRACE_HEADS, CAMODEL_TRACE_SPLITS,
# CAMODEL_TRACE_TIMEOUT_MINUTES, CAMODEL_TRACE_KEEP_WORK=1, BUILD_JOBS.
set -euo pipefail
umask 077

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
sim_lib="${toolkit}/tools/simulator/${soc}/lib"
msprof="${toolkit}/tools/profiler/bin/msprof"
config_source="${CAMODEL_CONFIG_PATH:-${sim_lib}}"
out="${1:-${repo}/reports/camodel_trace_$(date -u +%Y%m%dT%H%M%SZ)}"
if (( $# > 1 )); then echo "usage: $0 [new-output-directory]" >&2; exit 2; fi
for tool in cmake rsync python3 timeout; do command -v "${tool}" >/dev/null; done
for file in "${toolkit}/set_env.sh" "${msprof}" "${sim_lib}/libruntime_camodel.so" \
            "${config_source}/config.json" "${config_source}/config_stars.json"; do
    [[ -f "${file}" ]] || { echo "missing: ${file}" >&2; exit 2; }
done
# Never reuse an output directory: a stale trace must not satisfy this run.
mkdir -p "$(dirname "${out}")"
mkdir "${out}"
out="$(cd "${out}" && pwd)"
work="$(mktemp -d /tmp/ascend-camodel-trace.XXXXXX)"
cleanup() {
    local rc=$?
    if [[ "${CAMODEL_TRACE_KEEP_WORK:-0}" == 1 || ${rc} != 0 ]]; then
        echo "CAModel work retained: ${work} (exit ${rc})" >&2
    else
        rm -rf -- "${work}"
    fi
}
trap cleanup EXIT
mkdir "${work}/src" "${work}/config" "${work}/run" "${work}/profile"
rsync -a --exclude '.git' --exclude 'reports' --exclude 'build*' --exclude 'bin' \
    --exclude '__pycache__' --exclude 'sim_log' --exclude '*.dump' \
    --exclude 'profile_*.toml' --exclude 'probe/prof_output' "${repo}/" "${work}/src/"
cp "${config_source}/config.json" "${config_source}/config_stars.json" "${work}/config/"
cp -r "${work}/config" "${out}/config"
# Validate the overrides as JSON; do not add speculative 'profiling/format' keys.
python3 -m json.tool "${work}/config/config.json" >/dev/null
python3 -m json.tool "${work}/config/config_stars.json" >/dev/null
unset LD_PRELOAD LD_LIBRARY_PATH CAMODEL_CONFIG_PATH
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
# This script profiles the AIC decoder (--impl aic below), which the benchmark only
# contains when TQ_BENCH_LEGACY_DECODERS is ON. OFF is the default now that the AIV
# decoder is deleted and both benchmark targets isolate the fused operator.
cmake -S "${work}/src" -B "${work}/build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" -DASCEND_HOME_PATH="${toolkit}" \
    -DTQ_BENCH_LEGACY_DECODERS=ON \
    >"${out}/configure.log" 2>&1
cmake --build "${work}/build" --target bench_decode_splitk_sim bench_turboquant_decode_splitk \
    -j "${BUILD_JOBS:-4}" >"${out}/build.log" 2>&1
for target in bench_decode_splitk_sim bench_turboquant_decode_splitk; do
    cp "${work}/build/CMakeFiles/${target}.dir/flags.make" "${out}/${target}-flags.txt"
done
# Exactly one simulator family, without an inherited shim or PV-model directory.
export LD_LIBRARY_PATH="${work}/build/lib:${sim_lib}:${toolkit}/lib64:${toolkit}/devlib"
export CAMODEL_CONFIG_PATH="${work}/config"
page_tokens="${CAMODEL_TRACE_PAGE_TOKENS:-16}"
context="${CAMODEL_TRACE_CONTEXT:-${page_tokens}}"
heads="${CAMODEL_TRACE_HEADS:-1}"
splits="${CAMODEL_TRACE_SPLITS:-1}"
minutes="${CAMODEL_TRACE_TIMEOUT_MINUTES:-5}"
for value in "${context}" "${page_tokens}" "${heads}" "${splits}" "${minutes}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "expected positive integer: ${value}" >&2; exit 2; }
done
(( minutes <= 2880 )) || { echo "timeout must be <= 2880 minutes" >&2; exit 2; }
export CAMODEL_TRACE_APP="${work}/build/lib/bench_decode_splitk_sim"
export CAMODEL_TRACE_CONTEXT="${context}" CAMODEL_TRACE_HEADS="${heads}" CAMODEL_TRACE_SPLITS="${splits}"
export CAMODEL_TRACE_PAGE_TOKENS="${page_tokens}"
# A wrapper lets --application work on the installed 9.2 CLI without quoting
# application arguments into a shell string. It execs the production benchmark.
cat >"${work}/run/application.sh" <<'APP'
#!/usr/bin/env bash
set -euo pipefail
exec "${CAMODEL_TRACE_APP}" --impl aic --single --context "${CAMODEL_TRACE_CONTEXT}" \
    --page-tokens "${CAMODEL_TRACE_PAGE_TOKENS}" --heads "${CAMODEL_TRACE_HEADS}" --splitk "${CAMODEL_TRACE_SPLITS}"
APP
chmod 700 "${work}/run/application.sh"
"${msprof}" op simulator --help >"${out}/msprof-help.txt" 2>&1
{
    date -u +%FT%TZ
    uname -a
    printf 'toolkit=%s\nsoc=%s\nwork=%s\ncontext=%s\npage_tokens=%s\nheads=%s\nsplits=%s\n' \
        "${toolkit}" "${soc}" "${work}" "${context}" "${page_tokens}" "${heads}" "${splits}"
    printf 'LD_LIBRARY_PATH=%s\nCAMODEL_CONFIG_PATH=%s\n' "${LD_LIBRARY_PATH}" "${CAMODEL_CONFIG_PATH}"
    sha256sum "${CAMODEL_TRACE_APP}" "${work}/config/"*.json "${work}/src/tests/npu/bench/bench_turboquant_decode_splitk.cpp"
} >"${out}/environment.txt"
cd "${work}/run"
command=("${msprof}" op simulator --application="${work}/run/application.sh"
    --output="${work}/profile" --soc-version="${soc}" --aic-metrics=PipeUtilization
    --kernel-name=turboquant_paged_decode_splitk_aic --launch-count=1 --timeout="${minutes}")
printf '%q ' "${command[@]}" >"${out}/command.txt"
printf '\n' >>"${out}/command.txt"
echo "Profiling one launch; logs: ${out}/profile.log"
# The inner timeout is in minutes. The outer deadline also bounds export.
set +e
timeout --kill-after=30s "$((minutes * 60 + 120))s" "${command[@]}" >"${out}/profile.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" >"${out}/profile-exit-code.txt"
if (( rc != 0 )); then cat "${out}/profile.log" >&2; exit "${rc}"; fi
# msprof can finish without a matched kernel; require application success too.
if ! grep -Fq 'PASS: CPU references matched. Kernel launches: AIV=0 AIC=1.' "${out}/profile.log" ||
   ! grep -Fq 'Total tick' "${out}/profile.log"; then
    echo "Application validation or clean model shutdown missing; see ${out}/profile.log" >&2
    exit 1
fi
mapfile -d '' traces < <(find "${work}/profile" -path '*/simulator/trace.json' -type f -print0)
if (( ${#traces[@]} != 1 )); then
    echo "Expected one official aggregate simulator/trace.json, found ${#traces[@]}; see ${out}/profile.log" >&2
    exit 1
fi
# Validate first; copy the toolkit JSON byte-for-byte (including native lanes).
python3 "${repo}/scripts/validate_camodel_trace.py" "${traces[0]}" \
    --summary "${out}/trace-validation.json"
cp "${traces[0]}" "${out}/trace.json"
sha256sum "${out}/trace.json" >"${out}/trace.sha256"
printf '%s\n' "${traces[0]}" >"${out}/native-trace-path.txt"
echo "PASS: official trace exported and validated: ${out}/trace.json"
