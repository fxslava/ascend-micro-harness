#!/usr/bin/env bash
# The TurboQuant KV-cache write -> read round-trip attempt, and the two halves that
# stand in for it.
#
#   bash scripts/run_turboquant_e2e_roundtrip.sh            # host gates, then both halves
#   bash scripts/run_turboquant_e2e_roundtrip.sh host       # the two host gates only
#   bash scripts/run_turboquant_e2e_roundtrip.sh ingress    # the ingress CAModel canary only
#   bash scripts/run_turboquant_e2e_roundtrip.sh decode     # the fused-decode CAModel canaries only
#   bash scripts/run_turboquant_e2e_roundtrip.sh census     # re-collect evidence from existing runs
#
# WHY THERE IS NO SINGLE ROUND-TRIP RUN HERE
# ------------------------------------------
# `host` runs tests/cpu/test_turboquant_kv_layout_bridge.cpp, which maps every
# (block, kv_head, token, channel) of a block_size 64 geometry through BOTH address
# functions -- the ingress writers' STRIP contract and the fused decode's NZ-TILED
# NzTiledPackedByte -- and finds they name different bytes, with co-byte matchings that
# share not one pair. So there is no geometry at which an ingress store can be read back
# by the fused decode, and a genuine round trip is not expressible without changing one
# operator's layout. See reports/turboquant_e2e_roundtrip.md.
#
# What this script runs instead is the strongest available evidence: the two halves in
# isolation, each against its own oracle, at ONE tile. That is a canary, not a baseline.
# No ladder, no depth rung, no wrap, no multi-worker -- so nothing here supports a
# hardware claim under REFINEMENT_OF_COMBAT_KERNELS.md.
#
# Requires CANN (the container; the WSL host's CANN 8.5 cannot build these kernels).
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, E2E_ROOT, FUSED_DECODE_OPT_ROOT,
# SCATTER_SRC_DIR, SCATTER_BUILD_DIR.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
. "$(dirname "${BASH_SOURCE[0]}")/lib/work_root.sh"
root="${E2E_ROOT:-$(ascend_micro_work_dir ascend-tq-e2e-roundtrip)}"
mkdir -p "$root"
stage="${1:-all}"
printf 'Evidence root: %s\nStage: %s\n\n' "$root" "$stage"

# --------------------------------------------------------------------- evidence census
# The acceptance criteria, collected from a run directory rather than asserted: the PASS
# line, the simulator's tick count, how many of the per-core exception dumps are NON-EMPTY
# out of the total, and the three memory-fault patterns. Counts, not assertions, because
# "0 of 32" is evidence and "it did not assert" is not.
census() {
    local dir="$1" label="$2" log="${3:-}"
    printf '\n----- census: %s (%s)\n' "$label" "$dir"
    if [[ ! -d "$dir" ]]; then printf '  MISSING: no such run directory\n'; return 0; fi
    [[ -n "$log" ]] || log="$dir/run.log"
    if [[ -f "$log" ]]; then
        printf '  log: %s\n' "$log"
        { grep -E '(^|: )(PASS|FAIL)|PASS$|FAIL$' "$log" || true; } | sed 's/^/  /'
        { grep -E 'Total tick|Model RUN TIME' "$log" || true; } | sed 's/^/  /'
    else
        printf '  (no run log at %s)\n' "$log"
    fi
    local total empty nonempty
    total="$(find "$dir" -name 'core*.excp_log.dump' | wc -l)"
    nonempty="$(find "$dir" -name 'core*.excp_log.dump' -size +0 | wc -l)"
    printf '  exception dumps: %s of %s non-empty (core*.excp_log.dump)\n' "$nonempty" "$total"
    if ((nonempty)); then find "$dir" -name 'core*.excp_log.dump' -size +0 | sed 's/^/    /'; fi
    local pattern
    # `|| true` is load-bearing: `set -o pipefail` is on, and grep exits 1 when it finds
    # nothing -- which is the result we are here to record, not a failure.
    for pattern in 'invalid ldst addr' 'unrecognize ldst addr' 'su_ccu_mpu_err'; do
        local files lines
        files="$( { grep -rIl -- "$pattern" "$dir" 2>/dev/null || true; } | wc -l)"
        lines="$( { grep -rIc -- "$pattern" "$dir" 2>/dev/null || true; } |
            awk -F: '{ n += $NF } END { print n + 0 }')"
        printf '  %-26s %s match(es) in %s file(s)\n' "$pattern:" "$lines" "$files"
    done
}

# --------------------------------------------------------------------------- host gates
run_host() {
    mkdir -p "$repo/bin" "$root/host"
    printf '=== host gate 1/2: Tier-1 reshape_scatter oracle (the ingress contract)\n'
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$repo/include" \
        "$repo/tests/cpu/test_reshape_scatter_probe.cpp" -o "$repo/bin/test_reshape_scatter_probe"
    "$repo/bin/test_reshape_scatter_probe" 2>&1 | tee "$root/host/reshape_scatter_probe.log"
    printf '\n=== host gate 2/2: KV layout bridge -- is the round trip expressible at all?\n'
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$repo/include" \
        "$repo/tests/cpu/test_turboquant_kv_layout_bridge.cpp" \
        -o "$repo/bin/test_turboquant_kv_layout_bridge"
    "$repo/bin/test_turboquant_kv_layout_bridge" 2>&1 | tee "$root/host/kv_layout_bridge.log"
}

# ------------------------------------------------------- half 1: the ingress store (write)
# One tile: one token, one work item, grid 1, head 0 (the boundary head). The scatter
# runner rebuilds from scratch by design -- CANN's preprocess stage cannot run twice in one
# build directory -- so this is the long leg.
run_ingress() {
    export SCATTER_SRC_DIR="${SCATTER_SRC_DIR:-$root/scatter-src}"
    export SCATTER_BUILD_DIR="${SCATTER_BUILD_DIR:-$root/scatter-build}"
    printf '=== ingress half: scatter CAModel canary, head 0, 1 tile\n'
    bash "$repo/scripts/run_reshape_scatter_camodel.sh" --mode canary --head 0 2>&1 |
        tee "$root/ingress-canary.log"
    # The runner's ad-hoc argument path calls its audit with rows/runs/launches 0, which
    # makes the instruction census REPORT rather than ASSERT -- only the caller knows how
    # many rows the arguments imply. We do: --mode canary --head 0 is 1 row, 1 run, 1
    # launch. Re-audit the dumps it already captured at those counts, so the census is an
    # assertion. No second simulator run.
    local dumps="$SCATTER_BUILD_DIR/dumps_scatter_run"
    if [[ -f "$dumps/core0.veccore0.instr_popped_log.dump" ]]; then
        printf '\n=== ingress half: asserting re-audit at rows 1, runs 1, launches 1\n'
        python3 "$repo/scripts/analyze_scatter_trace.py" \
            "$dumps/core0.veccore0.instr_popped_log.dump" \
            "$dumps/core0.veccore0.instr_log.dump" \
            --aiv1 "$dumps/core0.veccore1.instr_popped_log.dump" \
            --rows 1 --runs 1 --launches 1 2>&1 |
            tee "$SCATTER_BUILD_DIR/audit_scatter_run_asserted.txt"
    else
        printf 'NOTE: no veccore0 dump at %s; the asserting re-audit cannot run\n' "$dumps"
    fi
}

# -------------------------------------------- half 2: the fused decode read (read)
# Two single-tile cases, no ladder rung: the isolated boundary logical page on the
# non-identity block table, and one tile of the real 64-token / 256-channel geometry.
# Against the fused decode's OWN host-built NZ-tiled fixture cache -- which is the whole
# point of this report: that cache is not what any ingress writer produces.
#
# FUSED_DECODE_OPT_ROOT should point at an existing build; `reuse` skips the ~40 min
# BiSheng pass. FUSED_DECODE_OPT_SKIP_NPU_GATE is an ENVIRONMENT VARIABLE read by that
# runner, not a cmake option, and it drops the RUN_MODE=npu strict-compile leg.
run_decode() {
    export FUSED_DECODE_OPT_ROOT="${FUSED_DECODE_OPT_ROOT:-$root/fused-decode}"
    export FUSED_DECODE_OPT_SKIP_NPU_GATE="${FUSED_DECODE_OPT_SKIP_NPU_GATE:-1}"
    local mode=reuse
    if [[ ! -x "$FUSED_DECODE_OPT_ROOT/build/lib/test_camodel_fused_decode_opt" ]]; then
        printf 'NOTE: no runner at %s/build/lib -- building from scratch\n' "$FUSED_DECODE_OPT_ROOT"
        mode=build-then-run
    fi
    printf '=== fused decode half: isolated logical page 0 (--canary 0)\n'
    if [[ "$mode" == reuse ]]; then
        FUSED_DECODE_OPT_TAG=e2e-isolated-page0 \
            bash "$repo/scripts/run_fused_decode_opt_camodel.sh" reuse --canary 0 2>&1 |
            tee "$root/decode-isolated-page0.log"
    else
        FUSED_DECODE_OPT_TAG=e2e-isolated-page0 \
            bash "$repo/scripts/run_fused_decode_opt_camodel.sh" --canary 0 2>&1 |
            tee "$root/decode-isolated-page0.log"
    fi
    printf '\n=== fused decode half: one full tile (--tiles 1 --tail-rows 64)\n'
    FUSED_DECODE_OPT_TAG=e2e-single-tile \
        bash "$repo/scripts/run_fused_decode_opt_camodel.sh" reuse --tiles 1 --tail-rows 64 2>&1 |
        tee "$root/decode-single-tile.log"
}

run_census() {
    printf '\n================ ACCEPTANCE EVIDENCE ================\n'
    local scatter_build="${SCATTER_BUILD_DIR:-$root/scatter-build}"
    local decode_root="${FUSED_DECODE_OPT_ROOT:-$root/fused-decode}"
    # The scatter runner archives each run's dumps under its own tag inside the build dir and
    # leaves the run log beside them as camodel_scatter_<tag>.log.
    local d tag
    shopt -s nullglob
    for d in "$scatter_build"/dumps_scatter_*; do
        tag="${d##*/dumps_scatter_}"
        census "$d" "ingress scatter canary, tag $tag" "$scatter_build/camodel_scatter_$tag.log"
        local audit
        for audit in "$scatter_build/audit_scatter_$tag.txt" \
                     "$scatter_build/audit_scatter_${tag}_asserted.txt"; do
            [[ -f "$audit" ]] || continue
            printf '  trace audit: %s\n' "$audit"
            { grep -E 'check 4|check 5|scatter trace audit' "$audit" || true; } | sed 's/^/    /'
        done
    done
    shopt -u nullglob
    census "$decode_root/e2e-isolated-page0" "fused decode, isolated logical page 0"
    census "$decode_root/e2e-single-tile" "fused decode, one full 64-row tile"
    printf '\nSOC reported by the model vs built for: grep halGetSocVersion in any run.log;\n'
    printf 'the model answers Ascend950PR_9589 while the build targets %s. Recorded, not chased.\n' "$soc"
}

case "$stage" in
    host)    run_host ;;
    ingress) run_ingress; run_census ;;
    decode)  run_decode;  run_census ;;
    census)  run_census ;;
    all)     run_host; run_ingress; run_decode; run_census ;;
    *) printf 'unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
