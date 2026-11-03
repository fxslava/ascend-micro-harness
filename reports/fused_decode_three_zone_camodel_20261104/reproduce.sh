#!/usr/bin/env bash
# Reproduces reports/fused_decode_three_zone_camodel_20261104.md on this host.
# Native WSL2 Ubuntu 22.04 + CANN 9.2.0-beta.2, no Docker. No physical NPU.
# Build/evidence root must be ext4, not the Windows mount.
set -uo pipefail

SRC=${SRC:-/mnt/c/Ascend/ascend-micro-harness}
R=${R:-/home/dmin/work/three-zone-check-20261103}
TOOLKIT=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}
export SOC_VERSION=${SOC_VERSION:-Ascend950PR_9599}

mkdir -p "$R/src" "$R/evidence"
rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' \
      --exclude 'sim_log' --exclude '__pycache__' "$SRC/" "$R/src/"

set +u; source "$TOOLKIT/set_env.sh"; set -u
export ASCEND_HOME_PATH="$TOOLKIT"
cd "$R/src"

# ---- section 2 host contract, section 1 models, section 5/6.6 auditor controls
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    tests/cpu/test_fused_decode_opt_contract.cpp -o "$R/evidence/host-contract"
"$R/evidence/host-contract"                        | tee "$R/evidence/host-contract.log"
python3 -u tests/cpu/test_fused_decode_opt_model.py | tee "$R/evidence/host-model.log"
python3 -u tests/sim/kernel/test_three_zone_schedule.py \
                                                   | tee "$R/evidence/host-three-zone.log"
python3 -u tests/cpu/test_fused_decode_opt_trace.py | tee "$R/evidence/host-audit-controls.log"
python3 -u scripts/audit_cross_core_fences.py       | tee "$R/evidence/host-fences.log"

# ---- device build: CAModel runner plus the production NPU object
cmake -S "$R/src" -B "$R/build" \
      -DASCEND_HOME_PATH="$TOOLKIT" -DSOC_VERSION="$SOC_VERSION" > "$R/evidence/configure.log" 2>&1
cmake --build "$R/build" \
      --target test_camodel_fused_decode_opt turboquant_fused_decode_opt_npu -j8 \
      > "$R/evidence/build.log" 2>&1 || { tail -60 "$R/evidence/build.log"; exit 1; }

# ---- sections 3/4 CAModel cases, each with its own section 5 trace audit
export FUSED_DECODE_OPT_RUNNER="$R/build/lib/test_camodel_fused_decode_opt"
run() { tag=$1; shift; bash "$R/src/scripts/run_fused_decode_opt_case.sh" "$R" "$tag" "$@"; }

# section 3: isolated logical-page boundaries, real geometry, nonzero pages
run iso-page0           --canary 0
run iso-page-mid        --canary 4
run iso-page-last       --canary 7
run iso-page-last-strip --canary 7 --tile 1

# section 4: short path, minimal deep path, wrap, odd, exact period, full depth
run depth-n3            --tiles 3
run deep-first-n4       --tiles 4
run wrap-n6             --tiles 6
run odd-n7              --tiles 7
run period-exact-n9     --tiles 9
run fulldepth-n12       --tiles 12

# This case is what exposed the auditor's one-directional id-7 allow-list:
# with the auditor as committed in 6b38819 the kernel passes and the trace
# audit aborts on the reserved ('SCALAR','VEC',7) BiSheng route. It passes
# end to end with the fix in scripts/analyze_fused_decode_opt_trace.py.
run tail-rows-n16       --tiles 16 --tail-rows 17

# Deferred in the 2026-11-04 session, listed so the gap is explicit:
#   run repeat-launch-n4x2   --tiles 4 --heads 17 --repeats 2
#   run empty-task           --empty 1
#   run workers-n16-k4       --tiles 16 --splits 4
#   run empty-worker-n6-k4   --tiles 6 --splits 4
#   run unmapped-hole-n12    --tiles 12 --hole 2
#   run varied-n9            --tiles 9 --varied 1
#   run varied-fulldepth-n12 --tiles 12 --varied 1
