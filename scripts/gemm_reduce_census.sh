#!/usr/bin/env bash
# The cross-core channel census and the per-engine instruction census for one evidence
# directory: what §5 asks a report to contain, read from the run's own traces.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_census.sh TAG [TAG...]
set -uo pipefail
root="${GEMM_EVIDENCE:-/work/evidence}"

for tag in "$@"; do
    printf '===== %s =====\n' "$tag"
    python3 - "$root/$tag/audit.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
print('status:', a['status'])
print('-- cross-core channels (ticks each wait blocked) --')
for name, ch in a['cross_core'].items():
    mark = ch.get('fault') or ch.get('note') or 'ok'
    print('  %-44s %-14s %s' % (name.replace('core0.', ''), mark, ch['stalls']))
print('-- per-engine instruction census --')
for c in a['cores']:
    name = c['core'].split('.')[1]
    vec = sum(v for k, v in c['opcodes'].items() if k.startswith('RV_'))
    print('  %-12s total=%-6d vector_ops=%-5d engines=%s'
          % (name, c['instructions'], vec, c['engines']))
PY
done
