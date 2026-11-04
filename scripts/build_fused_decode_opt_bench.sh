#!/usr/bin/env bash
# Build the shared Split-K benchmark with the KV4/FP8 leg, in both modes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${FUSED_OPT_BENCH_ROOT:-$(mktemp -d /tmp/ascend-fused-opt-bench.XXXXXX)}"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
printf 'Benchmark build root: %s\n' "$root"
mkdir -p "$root/src"
set +u
source "$toolkit/set_env.sh"
set -u
rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' "$repo/" "$root/src/"
cmake -S "$root/src" -B "$root/build" -DASCEND_HOME_PATH="$toolkit" >"$root/configure.log" 2>&1
cmake --build "$root/build" --target bench_decode_splitk_sim bench_turboquant_decode_splitk \
    test_fused_decode_opt_fixture -j4 >"$root/build.log" 2>&1 || { tail -80 "$root/build.log"; exit 1; }
"$root/build/lib/test_fused_decode_opt_fixture" | tee "$root/host-fixture.log"
printf 'PASS simulator + NPU benchmark builds: %s/build/lib\n' "$root"
