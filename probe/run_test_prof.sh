#!/usr/bin/env bash
set -euo pipefail

# --- Пути ---
TEST_BIN="${TEST_BIN:-/mnt/c/Ascend/ascend-micro-harness/build/test_camodel_turboquant_reshape_and_cache_a8}"
KERNEL_LIB_DIR="${KERNEL_LIB_DIR:-/mnt/c/Ascend/ascend-micro-harness/build/lib}"
SIM_LIB_DIR="${SIM_LIB_DIR:-/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/simulator/dav_3510/lib}"

# --- Параметры симуляции ---
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9589}"   # из вашего лога
OUTPUT_DIR="${OUTPUT_DIR:-./prof_output}"

# --- Проверки ---
die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$TEST_BIN" ]]    || die "нет теста: $TEST_BIN"
[[ -d "$SIM_LIB_DIR" ]] || die "нет каталога симулятора: $SIM_LIB_DIR"
command -v msprof >/dev/null 2>&1 || die "msprof не найден в PATH"

# --- Окружение для загрузки библиотек симулятора ---
export LD_LIBRARY_PATH="$SIM_LIB_DIR:$KERNEL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Запуск: подкоманда simulator, без --aic-metrics ---
echo "==> msprof op simulator"
echo "    soc-version : $SOC_VERSION"
echo "    application : $TEST_BIN"
echo "    output      : $OUTPUT_DIR"
echo

exec msprof op simulator \
    --soc-version="$SOC_VERSION" \
    --application="$TEST_BIN" \
    --output="$OUTPUT_DIR"