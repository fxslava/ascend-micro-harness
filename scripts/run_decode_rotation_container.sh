#!/usr/bin/env bash
# Ingress/egress rotation controls and the 64-block AIV grid: build and gate.
#
# Runs INSIDE ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12, where the
# toolkit and the Ascend950PR_9599 simulator live. The repository is expected to
# have been copied into the container's ext4 filesystem already -- never build
# out of a drvfs mount, because the page cache can hand BiSheng a stale copy of
# a source a Windows-side editor has just rewritten.
#
#   bash scripts/run_decode_rotation_container.sh strict   # -Wall -Werror, all legs
#   bash scripts/run_decode_rotation_container.sh sim      # the CAModel targets
#   bash scripts/run_decode_rotation_container.sh host     # host-only gates
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, ROT_BUILD_DIR.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
build_dir="${ROT_BUILD_DIR:-/work/tq-build-rot}"
mode="${1:-strict}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"

configure() {
    # Always a FRESH directory: CANN's preprocess stage links each device
    # object over itself, and a second pass in the same tree dies with
    # "unknown file type".
    rm -rf "${build_dir}"
    cmake -S "${repo_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
}

# The physical-device legs. They need no NPU to COMPILE, which is the whole of
# what a host without a 950PR can establish for them.
strict_targets=(
    turboquant_paged_decode_npu
    turboquant_single_token_decode_splitk_npu
    test_npu_decode_splitk_aic
    test_npu_decode_splitk_aiv
    test_npu_decode_single_token_unaligned
    bench_turboquant_decode_splitk
)

# The CAModel legs: the only ones that can RUN here.
sim_targets=(
    test_camodel_decode_splitk_boundary
    test_camodel_paged_decode_attention
    test_decode_splitk_aiv_sim
    test_decode_splitk_aic_sim
    test_decode_single_token_unaligned_sim
    bench_decode_splitk_sim
)

case "${mode}" in
host)
    for probe in single_token_decode splitk_decode aic_splitk_decode; do
        src="${repo_dir}/tests/cpu/test_${probe}_probe.cpp"
        [[ -f "${src}" ]] || continue
        g++ -std=c++17 -O2 -Wall -Wextra -Werror "${src}" -o "/tmp/${probe}_probe"
        "/tmp/${probe}_probe"
    done
    # The grid bound is a declaration the device guard and the host launcher
    # SHARE, so compiling it from a host translation unit is worth something
    # (REFINEMENT section 6.1); the egress oracles are tied to each other here
    # before either grades a device run.
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -I "${repo_dir}" \
        "${repo_dir}/tests/cpu/test_splitk_grid_bound.cpp" -o /tmp/grid_bound
    /tmp/grid_bound
    ;;
strict)
    configure
    cmake --build "${build_dir}" --target "${strict_targets[@]}" -j 4
    echo "PASS strict build: ${build_dir}"
    ;;
sim)
    configure
    cmake --build "${build_dir}" --target "${sim_targets[@]}" -j 4
    echo "PASS sim build: ${build_dir}"
    ;;
all)
    configure
    cmake --build "${build_dir}" --target "${strict_targets[@]}" "${sim_targets[@]}" -j 4
    echo "PASS build: ${build_dir}"
    ;;
*)
    echo "usage: $0 [strict|sim|all|host]" >&2
    exit 2
    ;;
esac
