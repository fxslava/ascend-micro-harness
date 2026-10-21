#!/usr/bin/env python3
"""Report dispatch and completion cycle span from native CAModel logs."""

import re
import sys
from pathlib import Path

TICK = re.compile(r"\[(\d+)\]")

def bounds(paths):
    ticks = [int(match.group(1)) for path in paths
             for line in Path(path).read_text(errors="replace").splitlines()
             if (match := TICK.search(line))]
    if not ticks:
        raise SystemExit(f"no cycle records in {paths}")
    return min(ticks), max(ticks), len(ticks)

dispatch = bounds(sys.argv[1::2])
complete = bounds(sys.argv[2::2])
print(f"CAModel cycles: dispatch {dispatch[0]}..{dispatch[1]} "
      f"({dispatch[1] - dispatch[0]} cycles, {dispatch[2]} instructions); "
      f"complete {complete[0]}..{complete[1]} "
      f"({complete[1] - complete[0]} cycles, {complete[2]} instructions)")
