#!/usr/bin/env bash
# The bounded canary ladder for the host-tiled fused decode.
#
# REFINEMENT_OF_COMBAT_KERNELS.md sections 3-5, in the five rungs a pipeline/metadata refactor of
# this operator has to clear before it is pushed or tagged for physical hardware. Every rung runs
# the production ComputeTask/Run*Pipeline over the production descriptor layout on CAModel, with
# every GM payload flooded with 0x7FC0BEEF, both exterior guard bands asserted, every input
# snapshotted before upload and compared bitwise after synchronisation, and the active rows
# compared bit-exactly against an FP64 host oracle.
#
#   rung 1  tiles = 1, tail = 0    isolated single tile, short/bounded dispatch
#   rung 2  tiles = 2, tail = 15   ragged boundary inside the bounded dispatch
#   rung 3  tiles = 5, tail = 1    one live row in the final tile, across page boundaries
#   rung 4  tiles = 3 = kSlots     the ring's priming depth exactly
#   rung 5  tiles = 6 = 2*kSlots   slot reuse and drain: wrap
#
# It deliberately runs NO long-context case. The simulator is cycle-accurate and a 16k-token
# context is hours of model time with nothing in it the wrap rung does not already exercise; the
# depth/wrap pair plus the three boundaries are the inductive argument (section 4), and the
# full-sequence regressions belong on silicon.
#
# Beyond the five rungs it also runs the isolated logical-page canaries (page 0, K/2, K-1 and the
# final 64-row tile of the final page) and the unmapped-page degradation rung, because those are
# the gates that would catch the closed-form cursor addressing the wrong page rather than the
# wrong number of rows.
#
#   bash scripts/run_fused_decode_canary_ladder.sh            # build, then run
#   bash scripts/run_fused_decode_canary_ladder.sh reuse      # reuse $FUSED_DECODE_OPT_ROOT
#
# FUSED_DECODE_OPT_ROOT   evidence/build root (a fresh ext4 mktemp dir by default)
# FUSED_DECODE_OPT_RUNNER an already-built runner to drive instead of building one
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${FUSED_DECODE_OPT_ROOT:-$(mktemp -d /tmp/fused-decode-canary.XXXXXX)}"
export FUSED_DECODE_OPT_ROOT="$root"
mkdir -p "$root"
printf 'canary ladder evidence root: %s\n' "$root"

if [[ "${1:-}" != reuse ]]; then
    # The Tier-1 host gates first: the launch/tiling contract, the capacity-one ring model, the
    # trace-auditor controls and the three-zone schedule model. Every CAModel case below compares
    # the device against oracles these check, so a failure here means the rest of this run could
    # not tell a wrong kernel from a wrong oracle.
    bash "$repo/scripts/run_fused_decode_opt_camodel.sh" build
fi
python3 "$repo/tests/sim/kernel/test_three_zone_schedule.py" | tee "$root/host-three-zone.log"

export FUSED_DECODE_OPT_RUNNER="${FUSED_DECODE_OPT_RUNNER:-$root/build/lib/test_camodel_fused_decode_opt}"
run() { bash "$repo/scripts/run_fused_decode_opt_case.sh" "$root" "$@"; }

# ---------------------------------------------------------------- section 3: isolated canaries
# One tile of the real 256-channel / 64-token geometry, on a non-identity block table (logical
# page p holds physical page 2p+1) with the full 18-page cache allocation kept, at logical pages
# 0, K/2 and K-1 for K = 8, plus the final 64-row tile of the final page. Rebasing the selected
# page to zero would not test the offsets, so none of these do.
run isolated-first --canary 0
run isolated-middle --canary 4
run isolated-last --canary 7
run isolated-last-tile --canary 7 --tile 1

# ------------------------------------------------------- the five bounded rungs, in order
run rung1-single-tile --tiles 1 --tail-rows 64
run rung2-ragged-tail --tiles 2 --tail-rows 15
run rung3-single-row-tail --tiles 5 --tail-rows 1
run rung4-ring-depth --tiles 3 --tail-rows 64
run rung5-ring-wrap --tiles 6 --tail-rows 64

# --------------------------------------------------- the closed-form cursor's own guard rails
# An unmapped logical page inside a live span is a geometry the closed-form tile cursor cannot
# express. The host builder refuses it (asserted in the Tier-1 contract gate); this plants one
# behind the builder's back and requires the kernel to skip the whole worker -- defined zero
# output, canaries intact, no deadlock on the cross-core flags -- rather than index through it.
run unmapped-degrades --tiles 12 --hole 2
# An empty worker: both roles must bypass the pipeline without touching a flag.
run empty-worker --empty 1
# Real Split-K workers, including an empty one, and repeated launches over one ragged task. Not
# among the five rungs, and required anyway: boundary isolation does not prove slot reuse across
# tasks, and a single-worker probe does not prove the tail worker's merge path.
run splitk-workers --tiles 6 --splits 4
run repeat-tasks --tiles 4 --heads 17 --repeats 2

printf 'PASS fused decode canary ladder (5 rungs + isolated canaries + workers): %s\n' "$root"
