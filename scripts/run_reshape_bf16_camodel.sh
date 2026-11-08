#!/usr/bin/env bash
# Build and run the TurboQuant BF16 prefill reshape-and-cache probe on CAModel
# over the inductive strip set, the same ladder the INT8 instance uses:
# 1 strip (cold start), 2 strips (two grid blocks, each addressing its own
# strip and its own physical page) and 3 strips (one interior step, which on a
# model with fewer AI Cores than strips also reuses a physical core and so
# catches a flag or token leaked by strip 0).
#
# There is no NPU on this host, so this is the verification gate. A pass means
# "executes on the cycle-accurate model over S = {0, 1, 2} and matches the
# audited golden", not a hardware claim.
#
# Wider contexts are ONE monolithic dispatch now. The kernel carries the
# inter-wave protocol -- boundary guard, bounded single-shot mode-4 channels,
# and an epilogue that drains every queue of every role into the scalar stream
# before the block retires -- so a grid wider than the model's 32 AI Cores
# spans several physical waves inside a single launch, which is the case that
# needs testing. --window is gone; a capped launch re-armed everything at each
# boundary and hid exactly this:
#
#   BF16_SLICES="1 32 33 48 64" scripts/run_reshape_bf16_camodel.sh
#
# BF16_SYNC selects the cross-core discipline: mode2 (the default and what
# ships) or mode4-single-shot (the fallback and bisection control, one strip per
# block). The recycled two-slot mode-4 ring that used to be the default is
# REFUTED and no longer reachable -- it failed this very gate at one strip, and
# reports/bf16_prefill_key_plane_fix.md has the traces. BF16_MODE4 is still read
# for compatibility: reuse now means mode2.
#
#   BF16_SYNC=mode4-single-shot BF16_SLICES=1 scripts/run_reshape_bf16_camodel.sh
#
# BF16_STRIPS_PER_BLOCK streams several strips through ONE grid block, over the
# mode-2 rendezvous, with one prime and one drain for the whole stream. That is
# the only rung on which a strip boundary is crossed inside a block, so it is
# the one that tests the loop rather than the grid. BF16_SLICES must be a
# multiple of it, and it needs mode2:
#
#   BF16_SLICES=8 BF16_STRIPS_PER_BLOCK=4 scripts/run_reshape_bf16_camodel.sh
#
# Audit the trace either way -- do not read the PASS line. Two runs of the
# recycled policy produced bit-exact output while their traces showed a second
# wait on a reused id passing before its set, and the third produced the wrong
# answer:
#
#   python3 scripts/audit_mode4_tick_order.py <rundir> --block 0 --cross
#
# Unlike the INT8 probe this kernel has no sparse boundary sampling: its grids
# are small enough that every block runs.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${BF16_BUILD_DIR:-$(mktemp -d /tmp/ascend-bf16-camodel-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"
slices="${BF16_SLICES:-1 2 3}"
strips_per_block="${BF16_STRIPS_PER_BLOCK:-1}"
# BF16_MODE4 is the old spelling. reuse named the refuted recycled ring, so it
# maps onto the discipline that replaced it rather than onto the fallback.
case "${BF16_MODE4:-}" in
    single-shot) sync_default=mode4-single-shot ;;
    *)           sync_default=mode2 ;;
esac
sync="${BF16_SYNC:-${sync_default}}"
case "${sync}" in
    mode2|mode4-single-shot) ;;
    *) echo "BF16_SYNC must be mode2 or mode4-single-shot, got: ${sync}" >&2; exit 2 ;;
esac

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN 9.2 toolkit not found: ${toolkit}" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${build}/lib:${sim}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

if [[ "${1:-all}" != "run-only" ]]; then
    if [[ -f "${build}/CMakeCache.txt" ]]; then
        echo "Use a fresh BF16_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    cmake -S "${repo}" -B "${build}" -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build}" \
        --target test_camodel_turboquant_reshape_and_cache_bf16 -j"${JOBS:-$(nproc)}"
fi

for n in ${slices}; do
    dir="${build}/s${n}_x${strips_per_block}_${sync}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        args=(--slices "${n}" --strips-per-block "${strips_per_block}")
        # Both directions named explicitly, so the shipped default can change
        # without this script quietly dispatching the other one.
        if [[ "${sync}" == "mode4-single-shot" ]]; then
            args+=(--mode4-single-shot)
        else
            args+=(--mode2)
        fi
        if ! timeout "${BF16_TIMEOUT:-1800s}" \
                "${build}/lib/test_camodel_turboquant_reshape_and_cache_bf16" \
                "${args[@]}" > run.log 2>&1; then
            cat run.log >&2
            echo "FAILED at ${n} slice(s)" >&2
            exit 1
        fi
        grep -E 'PASS|FAIL|Total tick|Model RUN TIME' run.log || true
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|su_ccu_mpu_err' run.log; then
            echo "CAModel errors in ${dir}" >&2
            exit 1
        fi
    )
done
echo "BF16 CAModel build and run directory: ${build}"
