#!/usr/bin/env bash
set -eo pipefail
harness=/mnt/d/Projects/ascend-micro-harness
scripts="$harness/scripts"
report="$harness/reports/sync_schedule_ablation/wave-v13-k"
root=/home/ladislav/ascend-sync-schedule-ablation
slot="$root/wave-v13-k"
sdk=/opt/Ascend92/cann-9.2.0-beta.2
sim="$sdk/tools/simulator/Ascend950PR_9599/lib"
build="$slot/build"
runner="$build/tests/test_fused_decode_e2e_sim"
manifest="$report/inputs.sha256"
source "$sdk/set_env.sh"
set -u
unset LD_PRELOAD LD_AUDIT CAMODEL_CONFIG_PATH
export LD_LIBRARY_PATH="$build/lib:$sim:$sdk/lib64:$sdk/devlib:$sdk/lib64/stub"

loader() {
    /lib64/ld-linux-x86-64.so.2 --list "$runner" | sed -E 's/[[:space:]]+\(0x[[:xdigit:]]+\)$//'
}

case "${1:-}" in
    package)
        [[ ! -e "$manifest" ]]
        grep -Fq 'PASS strict build' "$report/build-gates.txt"
        if grep -Eq 'warning:|error:' "$report/build-device.log" "$report/build-runner.log"; then
            printf 'Compiler diagnostics prevent packaging.\n' >&2
            exit 1
        fi
        if grep -Eiq 'printf|g_sysPrintFifoSpace|debug_print' "$report/device-symbols.txt"; then
            printf 'Device logging symbols prevent packaging.\n' >&2
            exit 1
        fi
        sha256sum --quiet -c "$report/../agent_02/wave-v11/inputs.sha256"
        loader >"$report/loader.txt"
        if grep -q 'not found' "$report/loader.txt"; then
            printf 'A runner dependency did not resolve.\n' >&2
            exit 1
        fi
        grep -q 'libpem_davinci.so => .*/Ascend950PR_9599/lib/' "$report/loader.txt"
        resolved=$(awk '$1=="libtq_decode_sim.so" && $2=="=>" {print $3}' "$report/loader.txt")
        [[ $(realpath "$resolved") == "$(realpath "$build/lib/libtq_decode_sim.so")" ]]
        staged=$(mktemp "$report/inputs.XXXXXX")
        {
            find "$slot/src" -type f
            printf '%s\n' "$runner" "$build/lib/libtq_decode_sim.so" \
                "$build/tq_decode_sim_aic_device_dir/device.o" "$build/tq_decode_sim_aiv_device_dir/device.o" \
                "$sdk/set_env.sh" "$sim/config.json" "$sim/config_stars.json" "$(command -v timeout)" \
                "$report/"{stage-k.py,build-k.sh,run-k.sh,candidate-k.patch,analyzer-review.json,analyzer-parent-findings.json,analyzer-findings.json,fence-audit.txt,loader.txt,device-symbols.txt,build-gates.txt,host-oracle.txt} \
                "$report/../agent_02/wave-v11/inputs.sha256" "$scripts/camodel_build_phase_guard.sh"
            awk '$2=="=>" && $3 ~ /^\// {print $3}' "$report/loader.txt"
        } | LC_ALL=C sort -u | xargs -d '\n' sha256sum >"$staged"
        sha256sum --quiet -c "$staged"
        mv -- "$staged" "$manifest"
        sha256sum "$manifest"
        ;;
    run)
        sha256sum --quiet -c "$manifest"
        stamp=$(sha256sum "$manifest" | cut -d ' ' -f 1)
        exec {ticket}>"$slot/run.lock"
        flock -n "$ticket"
        [[ ! -e "$report/launch-status.txt" && ! -e "$report/run.log" ]]
        printf 'APPROVED FOR CAMODEL %s\n' "$stamp" >"$report/approval.txt"
        status=0
        bash "$scripts/camodel_build_phase_guard.sh" camodel bash "$0" execute "$stamp" || status=$?
        printf '%s\n' "$status" >"$report/launch-status.txt"
        exit "$status"
        ;;
    execute)
        exec {lane0}>"$root/queue/camodel.0.lock"
        exec {lane1}>"$root/queue/camodel.1.lock"
        flock -x "$lane0"
        flock -x "$lane1"
        [[ $(sha256sum "$manifest" | cut -d ' ' -f 1) == "$2" ]]
        sha256sum --quiet -c "$manifest"
        cmp "$report/loader.txt" <(loader)
        mkdir "$slot/run"
        cd "$slot/run"
        invoke=(timeout --verbose --signal=KILL 150s env STAT.aicore_trace_enable=0 "$runner" --wave-v11-tiles 4)
        printf '%q ' "${invoke[@]}" >"$report/command.txt"
        printf '\n' >>"$report/command.txt"
        status=0
        /usr/bin/time -f '%e' -o "$report/wall-seconds.txt" "${invoke[@]}" >"$report/run.log" 2>&1 || status=$?
        (( status == 0 )) || exit "$status"
        [[ $(grep -Ec 'launch +N +208 .*grid 1 +tiles 4 +tail +16' "$report/run.log") == 1 ]]
        [[ $(grep -Fc 'BF16 oracle worst ULP 0.' "$report/run.log") == 1 ]]
        [[ $(grep -Fc 'PASS workspace telemetry: 6 per-head records, all other words intact' "$report/run.log") == 1 ]]
        [[ -z $(find . -name '*excp*.dump' -size +0c -print -quit) ]]
        grep -E '^\[INFO\] Total tick: [1-9][0-9]*$' "$report/run.log" | tee "$report/model-ticks.txt"
        [[ $(wc -l <"$report/model-ticks.txt") == 1 ]]
        ;;
    *) exit 2 ;;
esac
