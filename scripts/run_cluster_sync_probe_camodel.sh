#!/usr/bin/env bash
# Run tests/research/kernel_cluster_sync_probe.cpp on CAModel and read its
# stall census, for ONE question: does a cross-core wait block on EVERY use of
# an id, or only on its first?
#
#   bash scripts/run_cluster_sync_probe_camodel.sh [ITERS] [MODE]
#
# ITERS defaults to 4 and MODE to 2. Four iterations rather than two because
# MICROROUTINE_DESIGN_GUIDELINES.md section 6.2c's "mode-4 consumes" verdict
# came from two announcements per id, and
# reports/decode_splitk_aic_page_race.md section 4 measured the AIC -> AIV
# direction collapsing after the FIRST wait -- a behaviour two announcements
# cannot distinguish from a single blocking wait followed by a legitimately
# satisfied one.
#
# Read the trace, not the exit status. The probe moves no data, so a sticky
# latch retires every wait early and the run still finishes cleanly and still
# prints nothing wrong. Mode 4 lowers to set_intra_block / wait_intra_block and
# appears in the dumps as SET_INTRA_BLOCK / WAIT_INTRA_BLOCK; mode 2 lowers to
# ffts_cross_core_sync / wait_flag_dev and does not, so the verdict comes from
# the per-wait stall census over the common tick column.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, SYNC_PROBE_BUILD_DIR,
# SYNC_PROBE_SRC_DIR, SYNC_PROBE_TIMEOUT.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${SYNC_PROBE_SRC_DIR:-/tmp/ascend-sync-probe-src}"
build_dir="${SYNC_PROBE_BUILD_DIR:-/tmp/ascend-sync-probe-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
run_timeout="${SYNC_PROBE_TIMEOUT:-900s}"
iters="${1:-4}"
sync_mode="${2:-2}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
if [[ ! -f "${sim_lib_dir}/libnpu_drv_camodel.so" ]]; then
    echo "CAModel runtime not found: ${sim_lib_dir}" >&2
    exit 2
fi

set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"

# Build out of WSL ext4, never out of /mnt/c: the drvfs page cache can hand
# BiSheng a stale copy of a source a Windows-side editor has just rewritten.
# And always into a FRESH directory, because CANN's preprocess stage links each
# device object over itself and a second pass dies with "unknown file type".
mkdir -p "${src_dir}"
rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
    "${repo_dir}/" "${src_dir}/"
rm -rf "${build_dir}"
cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
    -DASCEND_HOME_PATH="${toolkit}"
cmake --build "${build_dir}" --target test_cluster_sync_probe_sim -j 4

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

# Each leg keeps its own evidence directory: the dumps land in the current
# directory and a second leg would otherwise overwrite the first's.
evidence="${build_dir}/ev_mode${sync_mode}_x${iters}"
mkdir -p "${evidence}"
cd "${evidence}"
rm -f core*.dump excp_log*.dump
timeout "${run_timeout}" "${build_dir}/lib/test_cluster_sync_probe_sim" \
    "${iters}" "${sync_mode}" 2>&1 | tee "probe_run.log"

echo
echo "=== per-wait stall census (the verdict) ==="
python3 "${repo_dir}/scripts/audit_mode4_tick_order.py" "${evidence}" \
    --block 0 --cross || true

echo
echo "=== nonempty exception dumps (must be none) ==="
find "${evidence}" -name "*excp*.dump" -size +0 -print | sed 's/^/    /' || true

echo
echo "evidence: ${evidence}"
