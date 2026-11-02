#!/usr/bin/env python3
"""Import a trace with Google's actual Perfetto engine and verify its lanes.

Supply --trace-processor /path/to/trace_processor, or --download to fetch the
official launcher from https://get.perfetto.dev/trace_processor. The launcher
caches its platform binary using Perfetto's normal download mechanism.
"""
import argparse
import csv
import io
from pathlib import Path
import subprocess
import tempfile
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--trace-processor", default="trace_processor")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="camodel-perfetto-") as temp:
        processor = args.trace_processor
        if args.download:
            launcher = Path(temp) / "trace_processor"
            with urllib.request.urlopen("https://get.perfetto.dev/trace_processor", timeout=60) as response:
                launcher.write_bytes(response.read())
            launcher.chmod(0o700)
            processor = str(launcher)
        # Import through the same engine used by ui.perfetto.dev, without
        # rewriting CANN's string process/thread ids or instruction timestamps.
        query = """
SELECT p.name AS core, t.name AS pipe, count(*) AS events,
       min(s.ts) AS first_ts_ns, max(s.ts+s.dur) AS last_end_ns
FROM slice s JOIN thread_track tt ON s.track_id=tt.id
JOIN thread t USING(utid) JOIN process p USING(upid)
WHERE s.dur > 0 GROUP BY core, pipe ORDER BY core, pipe
"""
        result = subprocess.run([processor, "-Q", query, str(args.trace.resolve())],
                                check=True, text=True, stdout=subprocess.PIPE)
        print(result.stdout, end="")
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        if len({row["core"] for row in rows}) < 3:
            parser.exit(1, "FAIL: Perfetto did not import all three MIX cores\n")
        pipes = {row["pipe"] for row in rows}
        if not {"MTE1", "MTE2", "MTE3", "CUBE", "FIXP"} <= pipes or not pipes & {"VECTOR", "RVECEX"}:
            parser.exit(1, "FAIL: Perfetto did not import all required pipe lanes\n")
        print("PASS: Perfetto imported native CAModel timestamps, cores and pipe lanes.")


if __name__ == "__main__":
    main()
