#!/usr/bin/env bash
# Name the faulting place in a gemm-reduce evidence directory: the distinct model errors, the
# intra-core HardEvent order per route, and the cross-core stall census.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_diagnose.sh TAG
set -uo pipefail
root="${GEMM_EVIDENCE:-/work/evidence}/$1"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== distinct model errors (count, text) ==="
grep -Eo '\[ERROR\].*' "$root/run.log" | sed 's/pc:0x[0-9a-f]*//' | sort | uniq -c | sort -rn | head -12

echo
echo "=== exception dumps ==="
for f in "$root"/*excp*.dump; do
    [ -e "$f" ] && printf '%s %s bytes\n' "$(basename "$f")" "$(stat -c%s "$f")"
done

echo
echo "=== intra-core place order and cross-core stalls ==="
PYTHONPATH="$src/scripts" python3 "$src/scripts/audit_mode4_tick_order.py" "$root" --cross \
    2>&1 | tail -60
