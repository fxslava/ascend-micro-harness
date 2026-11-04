#!/usr/bin/env bash
# The §5 instruction census, three legs over ONE geometry, from each run's own traces.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_compare.sh TOKENS HEADS [TAG]
#
# The three legs are the shipping vector reducer, the M = 1 Cube GEMV and the dense batched
# form, run at the same tokens/heads/splits so the engine counts are comparable. Ticks are
# printed because the runs are there; they are a simulator observation and not a throughput
# claim, and this is not an overlap measurement -- no intersecting per-engine interval export
# is made here, so no overlap is claimed.
set -uo pipefail
tokens="${1:-16}"
heads="${2:-4}"
suffix="${3:-cmp}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${GEMM_EVIDENCE:-/work/evidence}"

for leg in vector gemm batched; do
    bash "$src/scripts/gemm_reduce_run.sh" "$suffix-$leg" -- \
        --leg "$leg" --splits 8 --tokens "$tokens" --heads "$heads" --grid 1 >/dev/null 2>&1
done

python3 - "$root" "$suffix" "$tokens" "$heads" <<'PY'
import json, re, sys
root, suffix, tokens, heads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
print('%d tasks (%d tokens x %d heads), splits=8, one block' % (tokens * heads, tokens, heads))
print()
hdr = ('leg', 'aiv0', 'aiv1', 'aiv_total', 'aiv_worst', 'vec0', 'vec1', 'cube', 'fixp',
       'mte1', 'mte2(aic)', 'ticks')
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '|'.join(['---'] * len(hdr)) + '|')
for leg in ('vector', 'gemm', 'batched'):
    tag = '%s-%s' % (suffix, leg)
    try:
        a = json.load(open('%s/%s/audit.json' % (root, tag)))
    except Exception as exc:
        print('| %s | audit missing: %s |' % (leg, exc))
        continue
    by = {}
    for c in a['cores']:
        name = c['core'].split('.')[1]
        by[name] = (c['instructions'], c['engines'],
                    sum(v for k, v in c['opcodes'].items() if k.startswith('RV_')))
    v0, v1 = by.get('veccore0', (0, {}, 0)), by.get('veccore1', (0, {}, 0))
    cu = by.get('cubecore0', (0, {}, 0))
    # The last tick any core's trace reports: the three logs of a block share one tick column.
    ticks = 0
    for name in ('cubecore0', 'veccore0', 'veccore1'):
        try:
            with open('%s/%s/core0.%s.instr_popped_log.dump' % (root, tag, name)) as f:
                for line in f:
                    m = re.match(r'^\[\w+\]\s+\[(\d+)\]', line)
                    if m:
                        ticks = max(ticks, int(m.group(1)))
        except OSError:
            pass
    print('| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |' % (
        leg, v0[0], v1[0], v0[0] + v1[0], max(v0[0], v1[0]), v0[2], v1[2],
        cu[1].get('CUBE', 0), cu[1].get('FIXP', 0), cu[1].get('MTE1', 0),
        cu[1].get('MTE2', 0), ticks))
PY
