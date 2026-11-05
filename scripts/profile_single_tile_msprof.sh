#!/usr/bin/env bash
# One tile of the fused TurboQuant decode operator, under msprof's simulator mode, exported as
# a Chrome tracing JSON you can drop into Perfetto or chrome://tracing.
#
#   bash scripts/profile_single_tile_msprof.sh [tile|steady|reduce] [output-directory]
#
# WHY A SINGLE TILE. A CAModel run of a production-width launch takes tens of minutes and its
# trace is tens of megabytes, most of it the same six-phase body repeating. One tile is the
# smallest run in which every engine still appears: the MTE2 burst that reads the packed KV
# plane, the vector unpack and the L1 stage, the Cube's two Mmads (the score GEMM and the
# context GEMM), the Fixpipe landings, the online softmax and the exponent decay. That is
# enough to read instruction issue, operand latency and the handshake order off the timeline,
# at a size a browser can open.
#
# THE THREE PRESETS, and what each can and cannot show. Tiles are 64 rows (kCubeTileRows), so
# with 64-token pages the tile count IS the page count, which is how these are selected:
#
#   tile    1 tile.  ONE Cube score GEMM and ONE context GEMM, 16 query heads, D = 256, one
#           token, one KV head, no Split-K. numTiles < kSlots + 1, so the kernel takes its
#           BOUNDED SHORT pipeline -- this is the preset for reading one tile's dependence
#           chain end to end, and it CANNOT show steady-state overlap, because there is no
#           steady state at one tile. Default.
#   steady  4 tiles, the minimum deep path: the three-tile prologue, exactly ONE unrolled
#           steady phase, and the bounded three-step drain. This is the preset for the overlap
#           claim -- the score GEMM of tile k+1 against the softmax of tile k, the context GEMM
#           of k against the vector decay, and the MTE2 burst for k+3 against both.
#   reduce  2 tiles over 2 Split-K workers, so the REDUCTION PHASE runs as well. With
#           TQ_GEMM_REDUCE ON (the default) that is the Cube GEMM reduction out of the
#           attention phase's aliased UB pool, which is the one phase the other two presets
#           never reach: NeedsReduction() is false at one split.
#
# WHAT THIS IS NOT. Simulator ticks are not hardware time, and this script makes no speedup or
# overlap CLAIM -- it produces a timeline for a human to read. A measured overlap claim needs
# intersecting per-engine intervals, which is scripts/export_fused_decode_gantt.py's job, and
# a correctness claim needs the CAModel ladder (scripts/run_fused_decode_opt_camodel.sh) or the
# hardware ladder (tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp). There is no
# NPU on this host, so nothing here is a hardware number.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, BUILD_JOBS, PROFILE_TIMEOUT_MINUTES,
# PROFILE_KEEP_WORK=1, and PROFILE_CONTEXT / PROFILE_PAGE_TOKENS / PROFILE_SPLITS /
# PROFILE_HEADS / PROFILE_ROWS to override a preset's geometry outright.
set -euo pipefail
umask 077

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
sim_lib="$toolkit/tools/simulator/$soc/lib"
msprof="$toolkit/tools/profiler/bin/msprof"
kernel=turboquant_mm_fused_decode_opt_kv4fp8_half
minutes="${PROFILE_TIMEOUT_MINUTES:-15}"

preset="${1:-tile}"
case "$preset" in
    tile)   context=64;  page_tokens=64; splits=1;;
    steady) context=256; page_tokens=64; splits=1;;
    reduce) context=128; page_tokens=64; splits=2;;
    *) printf 'usage: %s [tile|steady|reduce] [output-directory]\n' "$0" >&2; exit 2;;
esac
context="${PROFILE_CONTEXT:-$context}"
page_tokens="${PROFILE_PAGE_TOKENS:-$page_tokens}"
splits="${PROFILE_SPLITS:-$splits}"
heads="${PROFILE_HEADS:-1}"      # KV heads
rows="${PROFILE_ROWS:-16}"       # GQA query heads per KV head: 16 query heads in total
out="${2:-$repo/reports/single_tile_msprof_${preset}_$(date -u +%Y%m%dT%H%M%SZ)}"

for value in "$context" "$page_tokens" "$splits" "$heads" "$rows" "$minutes"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'expected a positive integer: %s\n' "$value" >&2; exit 2; }
done
(( context % page_tokens == 0 )) || { printf 'context must be a multiple of page-tokens\n' >&2; exit 2; }
(( page_tokens == 64 || page_tokens == 128 )) ||
    { printf 'the fused operator takes 64- or 128-token pages\n' >&2; exit 2; }
(( minutes <= 240 )) || { printf 'timeout must be <= 240 minutes\n' >&2; exit 2; }
for tool in cmake rsync python3 timeout; do command -v "$tool" >/dev/null; done
for file in "$toolkit/set_env.sh" "$msprof" "$sim_lib/libruntime_camodel.so" \
            "$sim_lib/config.json" "$sim_lib/config_stars.json"; do
    [[ -f "$file" ]] || { printf 'missing: %s\n' "$file" >&2; exit 2; }
done

# Never reuse an output directory: a stale trace must not satisfy this run.
[[ ! -e "$out" ]] || { printf 'output already exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$(dirname "$out")"
mkdir "$out"
out="$(cd "$out" && pwd)"
work="$(mktemp -d /tmp/ascend-single-tile-msprof.XXXXXX)"
cleanup() {
    local rc=$?
    if [[ "${PROFILE_KEEP_WORK:-0}" == 1 || $rc != 0 ]]; then
        printf 'work retained: %s (exit %s)\n' "$work" "$rc" >&2
    else
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT
mkdir "$work/src" "$work/config" "$work/run" "$work/profile"

# An ext4 work copy, like every other device gate here: a drvfs page cache can hand BiSheng a
# stale copy of a source a Windows-side editor has just rewritten.
rsync -a --exclude '.git' --exclude 'reports' --exclude 'build*' --exclude 'bin' \
    --exclude '__pycache__' --exclude 'sim_log' --exclude '*.dump' \
    --exclude 'probe/prof_output' "$repo/" "$work/src/"
cp "$sim_lib/config.json" "$sim_lib/config_stars.json" "$work/config/"
cp -r "$work/config" "$out/config"
python3 -m json.tool "$work/config/config.json" >/dev/null
python3 -m json.tool "$work/config/config_stars.json" >/dev/null

unset LD_PRELOAD LD_LIBRARY_PATH CAMODEL_CONFIG_PATH
set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"

# The simulator twin of the decode benchmark. It links the CAModel build of the operator's
# kernel, which carries -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 whenever TQ_GEMM_REDUCE is ON --
# the default -- so the `reduce` preset profiles the Cube GEMM reduction. Configure this script
# with TQ_GEMM_REDUCE=OFF in CMAKE_EXTRA to profile the vector reducer instead.
cmake -S "$work/src" -B "$work/build" -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION="$soc" -DASCEND_HOME_PATH="$toolkit" ${CMAKE_EXTRA:-} \
    >"$out/configure.log" 2>&1
cmake --build "$work/build" --target bench_decode_splitk_sim -j "${BUILD_JOBS:-4}" \
    >"$out/build.log" 2>&1
bench="$work/build/lib/bench_decode_splitk_sim"
[[ -x "$bench" ]] || { printf 'the simulator benchmark did not build; see %s\n' "$out/build.log" >&2; exit 1; }
grep -E 'TQ_GEMM_REDUCE|TQ_BENCH_LEGACY_DECODERS' "$work/build/CMakeCache.txt" >"$out/cmake-switches.txt"

export LD_LIBRARY_PATH="$work/build/lib:$sim_lib:$toolkit/lib64:$toolkit/devlib"
export CAMODEL_CONFIG_PATH="$work/config"

# A wrapper, so --application needs no argument quoting on the 9.2 CLI. --single is what makes
# it ONE launch: warmup 0, iters 1, and no implicit K=1 baseline launch alongside it.
cat >"$work/run/application.sh" <<APP
#!/usr/bin/env bash
set -euo pipefail
cd "$work/run"
exec "$bench" --impl fused-opt --single --context $context --page-tokens $page_tokens \\
    --heads $heads --rows $rows --requests 1 --splitk $splits \\
    --csv-out "$out/results.csv"
APP
chmod 700 "$work/run/application.sh"
cp "$work/run/application.sh" "$out/application.sh"

{
    date -u +%FT%TZ
    uname -a
    printf 'git_commit='; git -C "$repo" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
    printf 'preset=%s\ncontext=%s\npage_tokens=%s\nsplits=%s\nkv_heads=%s\nquery_heads=%s\n' \
        "$preset" "$context" "$page_tokens" "$splits" "$heads" "$((heads * rows))"
    printf 'tiles_per_worker=%s\ntoolkit=%s\nsoc=%s\nkernel=%s\nwork=%s\n' \
        "$(( (context / page_tokens + splits - 1) / splits * (page_tokens / 64) ))" \
        "$toolkit" "$soc" "$kernel" "$work"
    printf 'LD_LIBRARY_PATH=%s\nCAMODEL_CONFIG_PATH=%s\n' "$LD_LIBRARY_PATH" "$CAMODEL_CONFIG_PATH"
    sha256sum "$bench" "$work/config/"*.json \
        "$work/src/kernels/ops/turboquant/turboquant_fused_decode_opt.cpp"
} >"$out/environment.txt"

cd "$work/run"
command=("$msprof" op simulator "--application=$work/run/application.sh"
    "--output=$work/profile" "--soc-version=$soc" --aic-metrics=PipeUtilization
    "--kernel-name=$kernel" --launch-count=1 --timeout="$minutes")
printf '%q ' "${command[@]}" >"$out/command.txt"
printf '\n' >>"$out/command.txt"
printf 'Profiling one launch of %s (%s preset); log: %s\n' "$kernel" "$preset" "$out/profile.log"
set +e
timeout --kill-after=30s "$((minutes * 60 + 120))s" "${command[@]}" >"$out/profile.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" >"$out/profile-exit-code.txt"
if (( rc != 0 )); then tail -60 "$out/profile.log" >&2; exit "$rc"; fi

# msprof can finish having matched no kernel at all, so require the application's own verdict:
# the benchmark validates every launch against its FP64 attention oracle before printing this.
if ! grep -Fq 'PASS: CPU references matched. Kernel launches: AIV=0 AIC=0 OPT=1.' "$out/profile.log"; then
    printf 'the profiled launch did not validate; see %s\n' "$out/profile.log" >&2
    exit 1
fi
if grep -Ei '\[error\]|\[aicore error\]|mismatch|FAIL:' "$out/profile.log"; then
    printf 'the profiled run reported an error; see %s\n' "$out/profile.log" >&2
    exit 1
fi

# The toolkit's own aggregate Chrome trace. Copied byte for byte, including its native lanes:
# CANN 9.2 puts the engine identity in the event's tid (CUBE, VEC, MTE1, MTE2, MTE3, FIX,
# SCALAR per core) and uses string pid/tid values, which is the schema Perfetto reads directly.
mapfile -d '' traces < <(find "$work/profile" -path '*/simulator/trace.json' -type f -print0)
if (( ${#traces[@]} != 1 )); then
    printf 'expected one simulator/trace.json, found %s; see %s\n' "${#traces[@]}" "$out/profile.log" >&2
    exit 1
fi
python3 "$repo/scripts/validate_camodel_trace.py" "${traces[0]}" \
    --summary "$out/trace-validation.json"
cp "${traces[0]}" "$out/msprof_trace.json"
sha256sum "$out/msprof_trace.json" >"$out/msprof_trace.sha256"
printf '%s\n' "${traces[0]}" >"$out/native-trace-path.txt"

printf '\nPASS: %s preset, one launch, trace exported and validated.\n' "$preset"
printf '  Chrome tracing JSON: %s\n' "$out/msprof_trace.json"
printf '  Open it in Perfetto (https://ui.perfetto.dev) or chrome://tracing. Engine lanes are\n'
printf '  the event tids: CUBE for the Mmads, FIX for the Fixpipe landings, MTE2 for the KV\n'
printf '  prefetch, MTE1/MTE3 for the L1 stage and the probability store, VEC/RVEC* for the\n'
printf '  softmax and the exponent decay.\n'
printf '  Lane census and schema check: %s\n' "$out/trace-validation.json"
printf '  Perfetto import check (optional, downloads trace_processor):\n'
printf '    python3 scripts/check_camodel_trace_perfetto.py %s --download\n' "$out/msprof_trace.json"
