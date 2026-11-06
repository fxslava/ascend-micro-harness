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
# BOTH ELEMENT TYPES. The operator is instantiated at fp16 and at bf16 (`--dtype`), and the whole
# ladder above runs for each. The two legs are NOT redundant: `scalar_t` reaches the output GM
# tensor, the UB row cast into it and -- on a Split-K launch -- the same pair inside the reducer,
# so the fp16 leg proves nothing about the bf16 rounding or the bf16 reducer write, and the bf16
# leg is run against its OWN correctly-rounded oracle at the same 0 ULP (the runner's comparison
# documents why bf16's coarser grid does not need a looser budget).
#
# Four cases run at fp16 only, deliberately rather than for time:
#
#   unmapped-degrades   asserts the closed-form cursor's density guard skips a worker. The guard
#   empty-worker        and the empty-worker bypass are properties of the tile cursor and the
#                       cross-core flag protocol, both of which are compiled from descriptor words
#   repeat-tasks        and ring indices with no scalar_t in them at all; `repeat-tasks` likewise
#                       exercises slot reuse across launches. Running them twice would re-test the
#                       same instructions. What IS element-type specific -- every GM write of a
#                       result -- is covered at both types by the rungs and the canaries, and the
#                       Split-K reducer's own write by splitk-workers at both types.
#
#   bash scripts/run_fused_decode_canary_ladder.sh            # build, then run both legs
#   bash scripts/run_fused_decode_canary_ladder.sh reuse      # reuse $FUSED_DECODE_OPT_ROOT
#
# FUSED_DECODE_OPT_ROOT   evidence/build root (a fresh ext4 mktemp dir by default)
# FUSED_DECODE_OPT_RUNNER an already-built runner to drive instead of building one
# FUSED_DECODE_OPT_DTYPES element types to run; default "fp16 bf16"
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
dtypes="${FUSED_DECODE_OPT_DTYPES:-fp16 bf16}"

for dtype in $dtypes; do
printf '\n================================ element type: %s\n' "$dtype"
run() { local tag="$1"; shift; bash "$repo/scripts/run_fused_decode_opt_case.sh" "$root" \
    "$dtype-$tag" --dtype "$dtype" "$@"; }

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

# Real Split-K workers, including an empty one. Not among the five rungs, and required at BOTH
# element types anyway: a single-worker probe does not prove the tail worker's merge path, and
# the merge path is where the reducer -- not the vector service -- performs the output cast and
# the output GM burst, which is the one piece of scalar_t code the rungs above never reach.
run splitk-workers --tiles 6 --splits 4
done

# ----------------------- element-type independent, run once (see the header note for why)
run_fp16() { bash "$repo/scripts/run_fused_decode_opt_case.sh" "$root" "$1" --dtype fp16 \
    "${@:2}"; }
# --------------------------------------------------- the closed-form cursor's own guard rails
# An unmapped logical page inside a live span is a geometry the closed-form tile cursor cannot
# express. The host builder refuses it (asserted in the Tier-1 contract gate); this plants one
# behind the builder's back and requires the kernel to skip the whole worker -- defined zero
# output, canaries intact, no deadlock on the cross-core flags -- rather than index through it.
run_fp16 unmapped-degrades --tiles 12 --hole 2
# An empty worker: both roles must bypass the pipeline without touching a flag.
run_fp16 empty-worker --empty 1
# Repeated launches over one ragged task: boundary isolation does not prove slot reuse across
# tasks.
run_fp16 repeat-tasks --tiles 4 --heads 17 --repeats 2

printf 'PASS fused decode canary ladder for [%s] (5 rungs + isolated canaries + Split-K workers '\
'each, cursor/flag rungs once): %s\n' "$dtypes" "$root"
