#!/usr/bin/env bash
# Print every SET_FLAG / WAIT_FLAG on one route of one core, with tick and PC, so a place whose
# order the audit rejects can be traced back to the instruction that issued it.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_place_trace.sh TAG CORE TRIGGER
#     CORE     cubecore0 | veccore0 | veccore1
#     TRIGGER  the TRIGGER PIPE to filter on, e.g. MTE1
set -uo pipefail
root="${GEMM_EVIDENCE:-/work/evidence}/$1"
core="${2:-cubecore0}"
trigger="${3:-MTE1}"
log="$root/core0.$core.instr_popped_log.dump"

echo "=== $core, TRIGGER PIPE:$trigger, in issue order (tick, PC, op, pipe, id) ==="
grep -E 'SET_FLAG|WAIT_FLAG' "$log" |
    grep -E "TRIGGER PIPE:$trigger," |
    sed -E 's/^\[[a-z]+\] \[0*([0-9]+)\].*\(PC: (0x[0-9a-f]+)\).*\(ID: [0-9]+\) +(SET_FLAG|WAIT_FLAG) +PIPE:([A-Z0-9]+), TRIGGER PIPE:([A-Z0-9]+), FLAG ID:(0x)?([0-9a-f]+).*/\1 \2 \3 \4->\5 id=\7/' |
    head -70

echo
echo "=== distinct PCs issuing SET_FLAG on this route ==="
grep -E 'SET_FLAG' "$log" | grep -E "TRIGGER PIPE:$trigger," |
    sed -E 's/.*\(PC: (0x[0-9a-f]+)\).*FLAG ID:(0x)?([0-9a-f]+).*/\1 id=\3/' |
    sort | uniq -c | sort -rn | head -20
