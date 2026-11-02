#!/usr/bin/env bash
# No NPU needed: test the actual benchmark against ACL test doubles.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
out="${1:-${repo}/reports/decode_benchmark_host_checks}"
mkdir -p "${out}"
work="$(mktemp -d /tmp/ascend-benchmark-host.XXXXXX)"
trap 'rm -rf -- "${work}"' EXIT
for mode in hardware simulator; do
    defs=()
    if [[ "${mode}" == simulator ]]; then defs=(-DASCEND_MICRO_BENCH_SIM); fi
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
        -Wl,--gc-sections -I"${toolkit}/include" "${defs[@]}" \
        "${repo}/scripts/test_decode_benchmark.cpp" -o "${work}/${mode}"
    "${work}/${mode}" "${out}/${mode}.csv" | tee "${out}/${mode}.txt"
done
