#!/usr/bin/env bash
# Four boundary/prefix oracles plus the mandatory isolated/depth/worker gates.
# Uses simulator libraries only; no physical device launch is performed.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${FUSED_DECODE_OPT_ROOT:-$(mktemp -d /tmp/fused-three-zone.XXXXXX)}"
export FUSED_DECODE_OPT_ROOT="$root"
if [[ "${1:-}" != reuse ]]; then
    bash "$repo/scripts/run_fused_decode_opt_camodel.sh" build
fi
python3 "$repo/tests/sim/kernel/test_three_zone_schedule.py" | tee "$root/host-three-zone.log"
export FUSED_DECODE_OPT_RUNNER="$root/build/lib/test_camodel_fused_decode_opt"
run() { bash "$repo/scripts/run_fused_decode_opt_case.sh" "$root" "$@"; }
# T0 plus the remaining logical-page boundaries run shared production math.
run T0 --canary 0
run isolated-middle --canary 4
run isolated-last --canary 7
run isolated-last-strip --canary 7 --tile 1
# Prefix oracles retain all intermediate tiles and real ring reuse. Skipping
# those transitions would invalidate a claim about the Period-6 pipeline.
run depth --tiles 3
run T1 --tiles 4
run wrap --tiles 6
run odd --tiles 7
run Tmid --tiles 9
run Tlast --tiles 16 --tail-rows 17
run repeat-tasks --tiles 4 --heads 17 --repeats 2
run empty --empty 1
run workers --tiles 16 --splits 4
run empty-worker --tiles 6 --splits 4
run unmapped --tiles 12 --hole 2
run varied --tiles 9 --varied 1
printf 'PASS three-zone verification: %s\n' "$root"
