"""Validate native CANN Chrome-trace output without inventing timeline events."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys


def validate(path):
    data = path.read_bytes()
    document = json.loads(data)
    events = document if isinstance(document, list) else document['traceEvents']
    if not isinstance(events, list) or not events:
        raise ValueError('empty/non-list native timeline')
    phases = Counter(e.get('ph', '?') for e in events)
    durations = [e for e in events if e.get('ph') == 'X']
    if not durations:
        raise ValueError('no complete native engine intervals')
    for event in durations:
        if not all(isinstance(event.get(k), (int, float)) and math.isfinite(event[k])
                   for k in ('ts', 'dur')) or event['dur'] < 0:
            raise ValueError('invalid native event time/duration')
    # CANN represents engines as tid lanes; categories can legitimately be absent.
    lanes = {(e.get('pid'), e.get('tid')): e.get('args', {}).get('name', '')
             for e in events if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    counts = defaultdict(Counter)
    for event in durations:
        counts[str(event.get('pid'))][lanes.get((event.get('pid'), event.get('tid')),
                                             str(event.get('tid')))] += 1
    lane_names = set(lanes.values()) | {str(e.get('tid')) for e in durations}
    if not any('CUBE' in s for s in lane_names) or not any('RVEC' in s for s in lane_names):
        raise ValueError(f'missing Cube/Vector engine lanes: {sorted(lane_names)}')
    return dict(status='PASS', sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                events=len(events), phases=dict(phases), positive_duration_events=sum(e['dur'] > 0 for e in durations),
                timestamp_range_us=[min(e['ts'] for e in durations), max(e['ts'] + e['dur'] for e in durations)],
                core_pipe_event_counts={k: dict(v) for k, v in counts.items()},
                note='Native CANN timeline, unchanged; simulator timestamps are not hardware timings.')


if __name__ == '__main__':
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
