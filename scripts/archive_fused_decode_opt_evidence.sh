#!/usr/bin/env bash
# Preserve a completed (or failed) ext4 build/run without copying its binaries.
set -euo pipefail
root="$(realpath "$1")"
out="$2"
[[ -f "$root/build.log" && -d "$root/src/kernels/ops/turboquant/fused_decode" ]]
mkdir -p "$out"
out="$(realpath "$out")"
for log in "$root"/*.log; do
    [[ ! -f "$log" ]] || cp "$log" "$out/$(basename "$log")"
done
printf '%s\n' "$root" > "$out/build-root.txt"
(
    cd "$root/src"
    find kernels/ops/turboquant/fused_decode -type f -print0
    printf '%s\0' kernels/ops/turboquant/turboquant_fused_decode_opt.cpp \
        routines/routine_vector_cube_softmax.hpp routines/routine_vector_online_softmax.hpp \
        routines/routine_turboquant_dequant4.hpp include/turboquant_fused_decode_opt.hpp \
        tests/camodel/test_camodel_fused_decode_opt.cpp
    for path in tests/npu/bench/bench_turboquant_decode_splitk.cpp \
        tests/npu/turboquant_fused_decode_opt_rig.hpp tests/common/fused_decode_opt_fixture.hpp \
        tests/cpu/test_fused_decode_opt_fixture.cpp cmake/SimTargets.cmake cmake/NpuTargets.cmake; do
        [[ ! -f "$path" ]] || printf '%s\0' "$path"
    done
) | (cd "$root/src"; xargs -0 sha256sum) > "$out/compiled-source-sha256.txt"
for target in bench_decode_splitk_sim bench_turboquant_decode_splitk; do
    for file in flags.make link.txt; do
        path="$root/build/CMakeFiles/$target.dir/$file"
        [[ ! -f "$path" ]] || cp "$path" "$out/$target-$file.txt"
    done
done
if [[ -d "$root/cli" ]]; then
    mkdir -p "$out/cli"
    cp "$root/cli/"*.txt "$out/cli/"
fi
for run in "$root"/*; do
    [[ -d "$run" && -f "$run/run.log" ]] || continue
    tag="$(basename "$run")"
    cp "$run/run.log" "$out/$tag-run.log"
    [[ ! -f "$run/audit.log" ]] || cp "$run/audit.log" "$out/$tag-audit.json"
    [[ ! -f "$run/audit.json" ]] || cp "$run/audit.json" "$out/$tag-audit.json"
    [[ ! -f "$run/results.csv" ]] || cp "$run/results.csv" "$out/$tag-results.csv"
    [[ ! -f "$run/command.txt" ]] || cp "$run/command.txt" "$out/$tag-command.txt"
    if compgen -G "$run/*.dump" >/dev/null; then
        (cd "$run"; tar -czf "$out/$tag-traces.tar.gz" ./*.dump)
    fi
done
printf 'Archived evidence: %s\n' "$out"
