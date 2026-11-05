#!/usr/bin/env python3
"""Export CAModel instruction traces as Gantt/Chrome-trace JSON, plus engine
occupancy and ACTUAL intersecting engine intervals.

Why this exists. `analyze_fused_decode_opt_trace.py` audits correctness of the
event protocol; `validate_fused_decode_opt_json_trace.py` validates msprof's
own op-level Chrome trace. Neither yields per-engine spans, so
REFINEMENT_OF_COMBAT_KERNELS.md section 5's requirement to "report engine
counts and actual intersecting MTE2/Vector intervals for an overlap claim" had
no tool behind it. This supplies the spans and the intersections.

Spans come from pairing each instruction id between the dispatch log
(`*.instr_popped_log.dump`, when the instruction was popped/issued) and the
completion log (`*.instr_log.dump`). The span is [dispatch tick, completion
tick]. CAModel omits some instructions from the completion log, so an unpaired
dispatch gets no span and is counted separately rather than given a fabricated
duration.

Units are MODEL TICKS, not nanoseconds and not silicon time. Chrome's trace
viewer labels the axis microseconds; the numbers here are ticks regardless, so
read relative structure, never absolute throughput.

Usage:
  export_fused_decode_gantt.py <case-dir> [--out gantt.json]
                               [--summary summary.json] [--core core0]
                               [--overlap MTE2,VEC] [--max-spans N]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_single_token_decode_trace import parse


def spans_for(popped_path):
    """Spans for one core's engine lanes, keyed by instruction id."""
    popped = parse(popped_path)
    completed = parse(popped_path.with_name(
        popped_path.name.replace('instr_popped_log', 'instr_log')))
    if not popped:
        return [], Counter()
    # Instruction ids are NOT unique within a dump: the model writes its dumps
    # when the process stops, so one dump covers every launch the runner made
    # and the id counter repeats. Pair occurrence-wise -- the k-th dispatch of
    # an id with the next completion of that id at or after it -- instead of
    # keying on the id alone, which would pair a dispatch with some other
    # launch's completion and yield negative durations.
    pending = defaultdict(list)
    for row in completed:
        pending[row['id']].append(row['tick'])
    for ticks in pending.values():
        ticks.sort(reverse=True)  # pop() takes the earliest remaining
    spans, unpaired = [], Counter()
    for row in sorted(popped, key=lambda r: r['tick']):
        ticks = pending.get(row['id'])
        while ticks and ticks[-1] < row['tick']:
            ticks.pop()  # a completion belonging to an unlogged dispatch
        if not ticks:
            unpaired[row['engine']] += 1
            continue
        finish = ticks.pop()
        spans.append(dict(id=row['id'], engine=row['engine'], mnemonic=row['mnemonic'],
                          start=row['tick'], end=finish, dur=finish - row['tick']))
    return spans, unpaired


def union_length(intervals):
    """Total length covered by a set of [start, end) intervals, merged."""
    merged, total = [], 0
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in merged:
        total += end - start
    return total, merged


def intersect_length(a, b):
    """Total length where merged interval set `a` overlaps merged set `b`."""
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case_dir', type=Path)
    ap.add_argument('--out', type=Path, help='Chrome-trace Gantt JSON path')
    ap.add_argument('--summary', type=Path, help='occupancy/overlap JSON path')
    ap.add_argument('--core', action='append', default=[],
                    help='restrict to cores whose dump name starts with this')
    ap.add_argument('--overlap', action='append', default=[],
                    help='engine pair to intersect, e.g. MTE2,RVECEX')
    ap.add_argument('--max-spans', type=int, default=400000,
                    help='refuse to emit a Gantt larger than this many spans')
    args = ap.parse_args()

    paths = sorted(args.case_dir.glob('core*.instr_popped_log.dump'))
    if not paths:
        raise SystemExit(f'no instruction traces in {args.case_dir}')
    if args.core:
        paths = [p for p in paths if any(p.name.startswith(c) for c in args.core)]
        if not paths:
            raise SystemExit('no traces matched --core')

    events, summary, total_spans = [], {}, 0
    for path in paths:
        spans, unpaired = spans_for(path)
        if not spans:
            continue  # an idle core in this grid; recorded by its absence
        core = path.name.split('.instr_popped_log')[0]
        total_spans += len(spans)
        by_engine = defaultdict(list)
        for s in spans:
            by_engine[s['engine']].append((s['start'], s['end'] if s['end'] > s['start']
                                           else s['start'] + 1))
        occupancy, merged = {}, {}
        for engine, intervals in by_engine.items():
            busy, runs = union_length(intervals)
            merged[engine] = runs
            occupancy[engine] = dict(spans=len(intervals), busy_ticks=busy,
                                     first_tick=min(i[0] for i in intervals),
                                     last_tick=max(i[1] for i in intervals))
        window = (min(s['start'] for s in spans), max(s['end'] for s in spans))
        pairs = {}
        requested = [tuple(p.split(',', 1)) for p in args.overlap] or \
                    [('MTE2', 'RVECEX'), ('MTE3', 'RVECEX'), ('CUBE', 'MTE1'),
                     ('MTE2', 'VEC'), ('CUBE', 'FIXP')]
        for left, right in requested:
            if left in merged and right in merged:
                pairs[f'{left}&{right}'] = intersect_length(merged[left], merged[right])
        summary[core] = dict(
            spans=len(spans), unpaired_dispatches=dict(unpaired),
            tick_window=list(window), span_ticks=window[1] - window[0],
            engines=occupancy, intersecting_ticks=pairs)
        # Chrome trace: one process per core, one thread lane per engine.
        events.append(dict(ph='M', name='process_name', pid=core, tid=0,
                           args=dict(name=core)))
        for engine in sorted(by_engine):
            events.append(dict(ph='M', name='thread_name', pid=core, tid=engine,
                               args=dict(name=engine)))
        for s in spans:
            events.append(dict(ph='X', name=s['mnemonic'], cat=s['engine'],
                               pid=core, tid=s['engine'], ts=s['start'],
                               dur=s['dur'] if s['dur'] > 0 else 1,
                               args=dict(instruction_id=s['id'], ticks=s['dur'])))

    if not summary:
        raise SystemExit('every selected core was idle; nothing to export')
    if total_spans > args.max_spans:
        raise SystemExit(f'{total_spans} spans exceeds --max-spans {args.max_spans}; '
                         'narrow with --core')

    document = dict(
        traceEvents=events, displayTimeUnit='ns',
        otherData=dict(
            source=str(args.case_dir),
            units='CAModel model ticks, NOT nanoseconds and NOT silicon time',
            spans_from='dispatch tick (instr_popped_log) to completion tick (instr_log)',
            caveat=('Unpaired dispatches get no span: CAModel omits some instructions '
                    'from the completion log. Overlap is measured on merged per-engine '
                    'interval unions, so it is real intersection, not source order.')))
    if args.out:
        args.out.write_text(json.dumps(document))
        print(f'gantt: {args.out} ({total_spans} spans, {len(summary)} active cores)')
    report = dict(cores=summary, totals=dict(spans=total_spans, active_cores=len(summary)),
                  note=document['otherData'])
    if args.summary:
        args.summary.write_text(json.dumps(report, indent=2))
        print(f'summary: {args.summary}')
    if not args.out and not args.summary:
        print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
