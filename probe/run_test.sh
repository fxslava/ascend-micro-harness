#!/usr/bin/env bash
set -euo pipefail

TEST_BIN="${TEST_BIN:-/mnt/c/Ascend/ascend-micro-harness/build/test_camodel_turboquant_reshape_and_cache_a8}"

# x86-64-симулятор из CANN
SIM_LIB_DIR="${SIM_LIB_DIR:-/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/simulator/dav_3510/lib}"
KERNEL_LIB_DIR="${KERNEL_LIB_DIR:-/mnt/c/Ascend/ascend-micro-harness/build/lib}"
DRIVER_LIB_DIR="${DRIVER_LIB_DIR:-/usr/local/Ascend/driver/lib64/driver}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$TEST_BIN" ]]    || die "нет теста: $TEST_BIN"
[[ -d "$SIM_LIB_DIR" ]] || die "нет каталога симулятора: $SIM_LIB_DIR"

export LD_LIBRARY_PATH="$SIM_LIB_DIR:$KERNEL_LIB_DIR:$DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$TEST_BIN" "$@"