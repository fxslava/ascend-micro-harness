#!/usr/bin/env bash
set -eo pipefail
harness=/mnt/d/Projects/ascend-micro-harness
report="$harness/reports/sync_schedule_ablation/wave-v13-k"
slot=/home/ladislav/ascend-sync-schedule-ablation/wave-v13-k
sdk=/opt/Ascend92/cann-9.2.0-beta.2
sim="$sdk/tools/simulator/Ascend950PR_9599/lib"
build="$slot/build"
source "$sdk/set_env.sh"
set -u
exec {ticket}>"$slot/build.lock"
flock -n "$ticket"
sha256sum --quiet -c "$report/../agent_02/wave-v11/inputs.sha256"
grep -Fq '"new_findings": 0' "$report/analyzer-review.json"
grep -Fq 'PASS 7 fenced announcements' "$report/fence-audit.txt"

# Keep strict warnings for generated wrappers by consuming the one unused
# SDK parameter. Kernel and fixture source retain the candidate-only diff.
strict_cmake=/home/ladislav/ascend-sync-schedule-ablation/production-wave-v13-phase0/src/cmake
cp "$strict_cmake/CannDeviceWrapperTargets.cmake" "$strict_cmake/CannDeviceWrapperSource.cmake" "$slot/src/cmake/"
cmake -S "$slot/src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=sim \
    -DTQ_BUILD_KERNELS=ON -DTQ_DECODE_TELEMETRY=ON -DSOC_VERSION=Ascend950PR_9599 -DASCEND_HOME_PATH="$sdk" \
    >"$report/configure.log" 2>&1
cmake --build "$build" --target tq_decode_sim -j3 >"$report/build-device.log" 2>&1
gmake -C "$build" turboquant_host/fast -j2 >"$report/build-runner.log" 2>&1
gmake -C "$build" test_fused_decode_e2e_sim/fast -j2 >>"$report/build-runner.log" 2>&1
! grep -Ei 'warning:|error:' "$report/build-device.log" "$report/build-runner.log"

export LD_LIBRARY_PATH="$build/lib:$sim:$sdk/lib64:$sdk/devlib:$sdk/lib64/stub"
runner="$build/tests/test_fused_decode_e2e_sim"
"$runner" --wave-v11-host-check 4 >"$report/host-oracle.txt"
grep -Fq 'HOST PASS native packed-input attention oracle at 0 BF16 ULP for 1536 outputs.' "$report/host-oracle.txt"
grep -Fq 'HOST PASS depth-2 geometry: grid=1, tiles=4, context=208, tail=16' "$report/host-oracle.txt"
aic="$build/tq_decode_sim_aic_device_dir/device.o"
aiv="$build/tq_decode_sim_aiv_device_dir/device.o"
"$sdk/x86_64-linux/ccec_compiler/bin/llvm-objdump" --syms "$aic" "$aiv" >"$report/device-symbols.txt"
[[ $(grep -c 'file format elf64-hiipu' "$report/device-symbols.txt") == 2 ]]
! grep -Eiq 'printf|g_sysPrintFifoSpace|debug_print' "$report/device-symbols.txt"
grep -Eq 'wave_v5_cube_softmax_two_strips_vf.*ILb1E.*\.vector.thread' "$report/device-symbols.txt"
cmp "$aic" /home/ladislav/ascend-sync-schedule-ablation/agent_02/wave-v11/build/tq_decode_sim_aic_device_dir/device.o
printf 'PASS strict build, host oracle, device symbols, and byte-identical AIC.\n' >"$report/build-gates.txt"
