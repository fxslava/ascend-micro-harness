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
    targets=(
        test_splitk_decode_probe
        test_aic_splitk_decode_probe
        camodel_single_token_decode_splitk_kernel
        camodel_paged_decode_attention_kernel
        test_camodel_decode_splitk_boundary
        bench_decode_splitk_sim
        test_decode_splitk_aiv_sim
        test_decode_splitk_aic_sim
    )
fi

echo "=== configure (${build_dir}) ==="
cmake -B "${build_dir}" -S . -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
configure_status=${PIPESTATUS[0]}
echo "configure status: ${configure_status}"
[ "${configure_status}" -eq 0 ] || exit "${configure_status}"

status=0
for target in "${targets[@]}"; do
    echo
    echo "=== build ${target} ==="
    cmake --build "${build_dir}" -j"$(nproc)" --target "${target}" 2>&1 \
        | grep -E "error:|warning:|Built target|Linking|static assertion" | tail -40
    rc=${PIPESTATUS[0]}
    echo "build status: ${rc}"
    [ "${rc}" -eq 0 ] || status="${rc}"
done

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
