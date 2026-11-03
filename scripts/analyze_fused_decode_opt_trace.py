"""Audit every core's dispatch/completion events and report engine census.

Mode-2 channel retirement is additionally checked with the shared cross-core
audit. Its missing scalar-release observations remain explicitly unmeasured.
"""
import json
from collections import Counter
from pathlib import Path
import sys

from analyze_single_token_decode_trace import parse, flag_key
from audit_mode4_tick_order import read, audit_stalls


def places(rows, order):
    marking, sets, waits = Counter(), Counter(), Counter()
    for row in sorted(rows, key=order):
        op = row['mnemonic']
        if op not in ('SET_FLAG', 'WAIT_FLAG'): continue
        key = flag_key(row)
        # CANN ArithProgression has __in_pipe__(S) __out_pipe__(V,S).
        # BiSheng emits reserved id 7 around those VF calls on either
        # direction of that one VEC/SCALAR pair; which directions appear
        # depends on how it schedules the body, so a long enough loop shows
        # both (N=12 emits V->S only, N=16 emits both). No user event
        # allocator owns this id, and no other pipe pair may carry it.
        # Still audit each route's full lifetime below.
        compiler_event = key in (('VEC', 'SCALAR', 7), ('SCALAR', 'VEC', 7))
        if '?' in key or not (0 <= key[2] <= 5 or compiler_event):
            raise ValueError(f'invalid/reserved user flag route {key}')
        if op == 'SET_FLAG':
            if marking[key]: raise ValueError(f'double SET {key} at ID {row["id"]}')
            marking[key] = 1; sets[key] += 1
        else:
            if not marking[key]: raise ValueError(f'WAIT without token {key} at ID {row["id"]}')
            marking[key] = 0; waits[key] += 1
    if sets != waits or any(marking.values()): raise ValueError('nonempty final marking')
    return sets


def audit(directory):
    paths = sorted(directory.glob('core*.instr_popped_log.dump'))
    if not paths: raise ValueError('missing instruction traces')
    for path in directory.glob('*excp*.dump'):
        if path.stat().st_size: raise ValueError(f'nonempty exception trace: {path}')
    cores = []
    for path in paths:
        popped = parse(path)
        completed = parse(path.with_name(path.name.replace('instr_popped_log', 'instr_log')))
        if not popped and not completed: continue
        if not popped or not completed: raise ValueError(f'incomplete trace: {path}')
        issued = places(popped, lambda r: r['id'])
        retired = places(completed, lambda r: (r['tick'], r['id']))
        if issued != retired: raise ValueError(f'dispatch/completion mismatch: {path}')
        census = Counter(row['mnemonic'] for row in popped)
        engines = Counter(row['engine'] for row in popped)
        cores.append(dict(core=path.name, instructions=len(popped), engines=dict(engines),
                          opcodes=dict(census), flags={str(k): v for k, v in issued.items()}))
    if not any(any(k.startswith('RV_') for k in c['opcodes']) for c in cores):
        raise ValueError('no Vector arithmetic; vacuous audit')
    cross, library_barriers = {}, {}
    blocks = sorted({int(p.name.split('.')[0][4:]) for p in paths})
    for block in blocks:
        _, waits, sets, _ = read(directory, block)
        # FFTS mode 2 sends AIC -> both AIVs and a two-AIV quorum -> AIC.
        # Channel 10 is used in both directions for the task join. The shared
        # audit's generic "all other announcers" rule cannot distinguish
        # those directions, so supply its actual producer set for each role.
        for cube in (False, True):
            consumers = [w for w in waits if w['engine'].startswith('cube') == cube]
            producers = {k: v for k, v in sets.items() if k[1].startswith('cube') != cube}
            channels, violations = audit_stalls(consumers, producers, None, set())
            if violations: raise ValueError('; '.join(violations))
            cross.update(channels)
        for (channel, producer), ticks in sets.items():
            # SyncAll<false>() uses CANN's reserved AIC global barrier
            # announcement (mode 0, id 11). It has no local WAIT_FLAG_DEV
            # counterpart; its wait is the toolkit's grid-wide GM protocol.
            if channel == 'm0:11' and producer == 'cubecore0':
                library_barriers[f'core{block}'] = len(ticks)
                continue
            if not channel.startswith('m2:') or not 0 <= int(channel[3:]) <= 10:
                raise ValueError(f'illegal cross-core channel: {channel}')
            consumers = ('veccore0', 'veccore1') if producer.startswith('cube') else ('cubecore0',)
            for consumer in consumers:
                count = sum(w['id'] == channel and w['engine'] == consumer for w in waits)
                if count != len(ticks): raise ValueError(f'cross-core census mismatch {channel}: {producer}->{consumer}')
    return dict(status='PASS', hardevents='capacity-one, balanced dispatch/completion, empty drain',
                cross_core=cross, cann_global_barrier_announcements=library_barriers, cores=cores)


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
