#!/usr/bin/env bash
# Strict build of the Split-K decode refactor inside the CANN 9.2 container.
#
# What this covers, and what it does not
# --------------------------------------
# This is the section 2 gate of REFINEMENT_OF_COMBAT_KERNELS.md -- the actual
# kernels compiled with -Wall -Werror for dav-3510, so every UB-arena and
# partition-layout static_assert is evaluated -- plus the host probes, which
# run here. It is NOT a CAModel run and says nothing about sections 3-5.
#
# From the repository root, with docker available:
#
#   docker run --rm -v "$(pwd)":/workspace -w /workspace \
#       ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 \
#       bash scripts/run_splitk_workspace_build.sh
#
# A single target can be selected as the first argument.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

build_dir="${SPLITK_BUILD_DIR:-build-splitk-workspace}"
targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
    # The AIV split-K kernel, its boundary runner and its validation twin
    # (camodel_single_token_decode_splitk_kernel, test_camodel_decode_splitk_boundary,
    # test_decode_splitk_aiv_sim) are deleted. The fused decode operator's strict compile
    # is camodel_fused_decode_opt_kernel, and its section 3-5 gates are
    # scripts/run_fused_decode_canary_ladder.sh.
    targets=(
        test_splitk_decode_probe
        test_aic_splitk_decode_probe
        camodel_paged_decode_attention_kernel
        camodel_fused_decode_opt_kernel
        bench_decode_splitk_sim
        test_decode_splitk_aic_sim
    )
fi

echo "=== configure (${build_dir}) ==="
cmake -B "${build_dir}" -S . -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
configure_status=${PIPESTATUS[0]}
echo "configure status: ${configure_status}"
[ "${configure_status}" -eq 0 ] || exit "${configure_status}"

# ONE invocation for every target, not one per target.
#
# Each ascendc_library in these targets' dependency chains is recompiled by CANN's
# BUILD_ALWAYS on every `cmake --build` invocation -- see the measured figures in the
# top-level CMakeLists.txt -- so the old per-target loop paid the full recompile of the
# shared kernel libraries once per target. On this target list that is the paged-decode
# and fused-decode operators rebuilt up to four times each for nothing. A single
# invocation pays it once and still fails the same way, because -Werror fails the build
# whichever target the diagnostic came from; the per-target retry below is only entered
# when something actually broke, so the slow path costs nothing on a green run.
echo
echo "=== build ${targets[*]} ==="
cmake --build "${build_dir}" -j"$(nproc)" --target "${targets[@]}" 2>&1 \
    | grep -E "error:|warning:|Built target|Linking|static assertion" | tail -60
status=${PIPESTATUS[0]}
echo "build status: ${status}"
if [ "${status}" -ne 0 ] && [ "${#targets[@]}" -gt 1 ]; then
    echo
    echo "=== the combined build failed; rebuilding target by target to name it ==="
    for target in "${targets[@]}"; do
        echo
        echo "=== build ${target} ==="
        cmake --build "${build_dir}" -j"$(nproc)" --target "${target}" 2>&1 \
            | grep -E "error:|warning:|Built target|Linking|static assertion" | tail -40
        echo "build status: ${PIPESTATUS[0]}"
    done
fi

echo
echo "=== host probes ==="
for probe in test_splitk_decode_probe test_aic_splitk_decode_probe; do
    if [ -x "${build_dir}/lib/${probe}" ]; then
        "./${build_dir}/lib/${probe}"
        rc=$?
        echo "${probe}: ${rc}"
        [ "${rc}" -eq 0 ] || status="${rc}"
    else
        echo "${probe}: SKIPPED (not built)"
    fi
done

echo
echo "overall status: ${status}"
exit "${status}"
