"""Read-only extraction of the final pair's scale preparation instructions."""
import sys
from pathlib import Path
sys.path.insert(0, '/mnt/d/Projects/ascend-micro-harness/scripts')
from analyze_single_token_decode_trace import parse

root = Path('/home/ladislav/ascend-sync-schedule-ablation/agent_02/wave-v11/run/t4.u97cm4')
for role in range(2):
    print('ROLE', role)
    for suffix in ('instr_popped_log.dump', 'instr_log.dump'):
        print(suffix)
        rows = parse(root / f'core0.veccore{role}.{suffix}')
        selected = [r for r in rows if 24689 <= r['tick'] < 24974 and r['engine'].startswith('RV')]
        print('last vector', max(r['tick'] for r in selected))
        print([r for r in selected if r['mnemonic'] in ('RV_VST','RV_VMULS')])
        print('first next vector', min(r['tick'] for r in rows if 24974 <= r['tick'] < 25366 and r['engine'].startswith('RV')))
        print('next writes', [r for r in rows if 24974 <= r['tick'] < 25366 and r['mnemonic']=='RV_VST'])
