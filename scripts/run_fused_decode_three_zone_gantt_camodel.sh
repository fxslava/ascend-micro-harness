#!/usr/bin/env bash
# Three-zone timeline capture: host gates -> strict build -> one case per zone
# structure -> event audit -> Gantt/Chrome-trace JSON + measured engine overlap.
# Optionally also captures a native msprof simulator trace. All outputs live in
# a fresh tree whose path is printed; nothing is written into the repository.
#
# This complements, and does not replace, run_fused_decode_three_zone_camodel.sh:
# that script is the correctness ladder (section 3 canaries, section 4 depth and
# wrap). This one exists for timeline/occupancy analysis, which section 5 asks
# for and which no previous script produced.
#
# Usage: run_fused_decode_three_zone_gantt_camodel.sh [all|build|gantt]
#   all    host gates, build, run every case, audit and export       (default)
#   build  host gates and build only
#   gantt  reuse an existing tree ($THREE_ZONE_GANTT_ROOT) and re-export
# Environment:
#   THREE_ZONE_GANTT_ROOT  reuse this tree instead of a fresh mktemp one
#   MSPROF_BENCH_ROOT      a build root from build_fused_decode_opt_bench.sh;
#                          when set, also capture the native msprof trace
#   THREE_ZONE_TIMEOUT     per-case timeout, default 1800s
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
mode="${1:-all}"
[[ "$mode" == all || "$mode" == build || "$mode" == gantt ]] ||
    { echo "usage: $0 [all|build|gantt]" >&2; exit 2; }

if [[ "$mode" == gantt ]]; then
    work_dir="${THREE_ZONE_GANTT_ROOT:?gantt mode needs THREE_ZONE_GANTT_ROOT}"
    [[ -d "$work_dir" ]] || { echo "no such tree: $work_dir" >&2; exit 2; }
else
    work_dir="${THREE_ZONE_GANTT_ROOT:-$(mktemp -d /tmp/ascend-three-zone-gantt.XXXXXX)}"
fi
mkdir -p "$work_dir"
printf 'artifact directory: %s\n' "$work_dir"

# One case per zone structure the schedule can take. The tile count is what
# selects the structure (kSlots=3, kPhases=6, steady_limit=numTiles-3), so these
# are the classes, not an arbitrary ladder. Tiles are 64 tokens each.
#   tag                tiles  structure
#   short-n3             3     bounded short path, no steady loop at all
#   deep-noperiod-n4     4     deep path, residue only, no complete period
#   deep-exact-n9        9     prologue + exactly one Period-6 + drain
#   deep-ragged-n12     12     one full period + residue-3 steady tail + drain
#   deep-twoperiod-n16  16     two full periods + residue-1, partial last tile
cases=(short-n3:3: deep-noperiod-n4:4: deep-exact-n9:9: deep-ragged-n12:12:
       deep-twoperiod-n16:16:--tail-rows,17)

if [[ "$mode" != gantt ]]; then
    # ---- host gates. Cheap, and they fail before any simulator time is spent.
    g++ -std=c++17 -O2 -Wall -Wextra -Werror \
        "$repo_dir/tests/cpu/test_fused_decode_opt_contract.cpp" -o "$work_dir/host-contract"
    "$work_dir/host-contract" | tee "$work_dir/host-contract.log"
    python3 -u "$repo_dir/tests/sim/kernel/test_three_zone_schedule.py" \
        | tee "$work_dir/host-three-zone.log"
    python3 -u "$repo_dir/tests/cpu/test_fused_decode_opt_trace.py" \
        > "$work_dir/host-audit-controls.log" 2>&1
    python3 -u "$repo_dir/scripts/audit_cross_core_fences.py" \
        > "$work_dir/host-fences.log" 2>&1

    set +u
    source "$toolkit/set_env.sh"
    set -u
    export ASCEND_HOME_PATH="$toolkit"
    mkdir -p "$work_dir/src"
    if command -v rsync >/dev/null; then
        rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' \
            --exclude 'sim_log' "$repo_dir/" "$work_dir/src/"
    else
        tar -C "$repo_dir" --exclude='.git' --exclude='reports' --exclude='bin' \
            --exclude='build*' --exclude='sim_log' -cf - . | tar -C "$work_dir/src" -xf -
    fi
    cmake -S "$work_dir/src" -B "$work_dir/build" -G "Unix Makefiles" \
        -DSOC_VERSION="$soc" -DASCEND_HOME_PATH="$toolkit" > "$work_dir/configure.log" 2>&1
    cmake --build "$work_dir/build" --target test_camodel_fused_decode_opt -j"${JOBS:-8}" \
        > "$work_dir/build.log" 2>&1 || { tail -60 "$work_dir/build.log" >&2; exit 1; }
    [[ "$mode" == build ]] && { printf 'PASS build only: %s\n' "$work_dir"; exit 0; }
fi

set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"
sim="$toolkit/tools/simulator/$soc/lib"
export LD_LIBRARY_PATH="$work_dir/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
runner="$work_dir/build/lib/test_camodel_fused_decode_opt"
[[ -x "$runner" ]] || { echo "missing runner: $runner" >&2; exit 2; }

: > "$work_dir/summary.txt"
for spec in "${cases[@]}"; do
    tag="${spec%%:*}"; rest="${spec#*:}"; tiles="${rest%%:*}"; extra="${rest#*:}"
    dir="$work_dir/$tag"
    if [[ "$mode" != gantt ]]; then
        mkdir -p "$dir"
        read -r -a extra_args <<< "${extra//,/ }"
        printf '%q ' "$runner" --tiles "$tiles" "${extra_args[@]}" > "$dir/command.txt"
        printf '\n' >> "$dir/command.txt"
        status=0
        (cd "$dir"; timeout "${THREE_ZONE_TIMEOUT:-1800s}" "$runner" --tiles "$tiles" \
            "${extra_args[@]}" > run.log 2>&1) || status=$?
        if ((status)); then tail -40 "$dir/run.log" >&2; echo "FAIL $tag (exit $status)" >&2; exit 1; fi
        if grep -Eiq '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$dir/run.log"; then
            tail -40 "$dir/run.log" >&2; echo "FAIL $tag (log)" >&2; exit 1
        fi
        grep -q 'Model stopped successfully' "$dir/run.log" ||
            { echo "FAIL $tag: model did not stop cleanly" >&2; exit 1; }
        # Correctness of the event protocol first: a timeline from a run whose
        # handshakes did not audit is not worth analysing.
        python3 "$repo_dir/scripts/analyze_fused_decode_opt_trace.py" "$dir" > "$dir/audit.json"
    fi
    # Timeline. Restricted to block 0, the only block a single-task grid runs;
    # every other block's dumps are a scalar prologue and an early return, and
    # 32 blocks of them would bloat the Gantt without adding a span of work.
    python3 "$repo_dir/scripts/export_fused_decode_gantt.py" "$dir" --core core0 \
        --out "$dir/gantt.json" --summary "$dir/gantt-summary.json" > "$dir/gantt.log"
    ticks="$(grep -oP 'Total tick: \K[0-9]+' "$dir/run.log" | tail -1)"
    printf 'PASS %-20s tiles=%-3s ticks=%-8s %s\n' "$tag" "$tiles" "${ticks:-NA}" \
        "$(sed -n 1p "$dir/gantt.log")" | tee -a "$work_dir/summary.txt"
done

# Cross-case occupancy and MEASURED engine intersection, in model ticks.
python3 - "$work_dir" <<'PY' | tee "$work_dir/overlap.txt"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f'{"case":22s} {"core":16s} {"span_ticks":>10s} {"MTE3&RVECEX":>12s} '
      f'{"MTE2&RVECEX":>12s} {"MTE2&VEC":>9s} {"CUBE&FIXP":>9s} {"unpaired":>9s}')
for summary in sorted(root.glob('*/gantt-summary.json')):
    d = json.loads(summary.read_text())
    for core, c in sorted(d['cores'].items()):
        pairs = c['intersecting_ticks']
        print(f'{summary.parent.name:22s} {core:16s} {c["span_ticks"]:10d} '
              f'{pairs.get("MTE3&RVECEX", 0):12d} {pairs.get("MTE2&RVECEX", 0):12d} '
              f'{pairs.get("MTE2&VEC", 0):9d} {pairs.get("CUBE&FIXP", 0):9d} '
              f'{sum(c["unpaired_dispatches"].values()):9d}')
print()
print('Ticks are CAModel model ticks, not nanoseconds and not silicon time.')
print('Intersections are measured on merged per-engine interval unions, so a')
print('nonzero value is real concurrency in the model -- not source order.')
print('Unpaired dispatches have no completion line in the dump and no span.')
PY

# Optional native msprof simulator capture, reusing the existing entry point so
# there is one definition of how this repository drives msprof.
if [[ -n "${MSPROF_BENCH_ROOT:-}" && "$mode" != gantt ]]; then
    printf 'native msprof capture from bench root %s\n' "$MSPROF_BENCH_ROOT"
    FUSED_OPT_TRACE_OUT="$work_dir/msprof-fused-opt" \
        bash "$repo_dir/scripts/capture_fused_decode_opt_camodel_json.sh" \
        "$MSPROF_BENCH_ROOT" fused-opt 768 | tee "$work_dir/msprof.log"
else
    printf 'native msprof capture: SKIPPED (set MSPROF_BENCH_ROOT to enable)\n' \
        | tee -a "$work_dir/summary.txt"
fi

printf 'PASS three-zone timeline capture; evidence: %s\n' "$work_dir"
