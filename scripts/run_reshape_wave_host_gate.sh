#!/usr/bin/env bash
# Host-only gate for the MIX_AIC_1_2 inter-wave protocol
# (MICROROUTINE_DESIGN_GUIDELINES.md section 6.5,
#  REFINEMENT_OF_COMBAT_KERNELS.md section 6).
#
# This is what can be established on a machine with no CANN toolkit and no
# Ascend950PR_9599 simulator. It is NOT the CAModel gate and it is not a
# hardware claim: it compiles the protocol arithmetic the kernels are built on,
# audits the cross-core id census for both channel policies, audits the scalar
# fence in front of every cross-core announcement, compiles every host harness
# strictly, and explores the token/queue model of a physical block running two
# grid blocks back to back.
#
# What it does NOT cover, and what still has to run on a machine that has the
# toolkit: device compilation of the two kernels (BiSheng, CANN 9.2), the
# isolated boundary canaries, the depth and wrap tests, the wave ladder on
# CAModel or silicon, and the trace audit. Record those separately.
#
#   bash scripts/run_reshape_wave_host_gate.sh
#
# Needs g++ with C++17 and python3. The ACL declarations the NPU harnesses
# include come from tests/host_stubs, which declares the surface they use and
# links to nothing: a pass means "the host code compiles clean", never "the
# harness works".
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stub="${repo}/tests/host_stubs"
out="${WAVE_GATE_DIR:-$(mktemp -d /tmp/ascend-wave-host-gate-XXXXXX)}"
# mktemp creates its directory; a caller-supplied WAVE_GATE_DIR may not exist,
# and every stage below writes a binary or a compiler log into it. Without this
# the gate reports FAIL for each of them with "No such file or directory",
# which looks exactly like a real compile failure.
mkdir -p "${out}"
cd "${repo}"
status=0

echo "host gate output directory: ${out}"
echo

echo "=== 1. protocol header: static_asserts, id census, boundary guard ==="
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. \
       tests/cpu/test_wave_protocol_header.cpp -o "${out}/test_wave_protocol_header"; then
    "${out}/test_wave_protocol_header" || status=1
else
    echo "FAIL compile tests/cpu/test_wave_protocol_header.cpp"
    status=1
fi

echo
# The precondition the recycled default rests on, and the one no other
# gate here can see: guidelines 6.2a and 6.2c. A missing drain balances
# every census and can still produce bit-exact output while a landing
# panel is read a generation early.
echo "=== 2. cross-core announcement fences ==="
python3 scripts/audit_cross_core_fences.py \
    kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp \
    kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp \
    tests/research/kernel_cluster_sync_probe.cpp || status=1

echo
echo "=== 3. host harnesses, -Wall -Wextra -Werror ==="
for f in tests/npu/validation/test_turboquant_a8_validation.cpp \
         tests/npu/validation/test_turboquant_bf16_validation.cpp \
         tests/npu/bench/bench_turboquant_reshape_and_cache.cpp \
         tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp \
         tests/camodel/test_camodel_turboquant_reshape_and_cache_bf16.cpp \
         tests/research/test_cluster_sync_probe.cpp; do
    printf '%-62s ' "${f}"
    if g++ -std=c++17 -Wall -Wextra -Werror -I"${stub}" -I. -fsyntax-only "${f}" \
           2> "${out}/gate.err"; then
        echo OK
    else
        echo FAIL
        sed -n '1,40p' "${out}/gate.err"
        status=1
    fi
done

echo
echo "=== 4. Tier-1 host probe: the operator's arithmetic contract is unchanged ==="
printf '%-62s ' tests/cpu/test_reshape_and_cache_a8_probe.cpp
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
       tests/cpu/test_reshape_and_cache_a8_probe.cpp \
       -o "${out}/test_reshape_and_cache_a8_probe" 2> "${out}/gate.err"; then
    echo OK
    "${out}/test_reshape_and_cache_a8_probe" || status=1
else
    echo FAIL
    sed -n '1,40p' "${out}/gate.err"
    status=1
fi

echo
echo "=== 5. token/queue models ==="
python3 tests/cpu/test_reshape_wave_model.py || status=1
echo
python3 tests/cpu/test_decode_ring_model.py | tail -4 || status=1

echo
if [[ "${status}" == 0 ]]; then
    echo "HOST GATE PASS -- see the header of this script for what it does not cover"
else
    echo "HOST GATE FAIL"
fi
exit "${status}"
