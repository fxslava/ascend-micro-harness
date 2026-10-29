#!/usr/bin/env python3
"""Verify completed baseline/regression artifacts and emit a compact manifest.

Does not launch a kernel. Rechecks all event traces, exception dumps and strict
runner logs, so missing, partial or failed runs cannot become a PASS manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from analyze_decode_ring_trace import audit

BASELINE = [(f"{impl}_{tag}", strips) for impl in ("aiv", "splitk")
            for tag, strips in (("head", 1), ("middle", 1), ("tail", 1), ("depth", 2), ("wrap", 4))]
REGRESSIONS = [("regression_rows6", 1), ("regression_pages2", 4),
               ("regression_odd3", 3), ("regression_splitk2", None),
               ("regression_splitk4", None)]
ERROR = re.compile(r"\[(?:error|fatal)\]|\bFAIL\b|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err", re.I)


def verify(root, name, strips):
    directory = root / name
    log = (directory / "run.log").read_text(errors="replace")
    if ERROR.search(log) or "Model stopped successfully" not in log:
        raise ValueError(f"failed/incomplete model run: {name}")
    if name.startswith("aiv_") and "byte-exact boundary oracle checked" not in log:
        raise ValueError(f"missing exact boundary check: {name}")
    if name.startswith("splitk_") and "byte-exact context; ASSERT_CANARY_INTACT" not in log:
        raise ValueError(f"missing exact/isolation check: {name}")
    if name.startswith("regression_splitk"):
        if "PASS: CPU references matched." not in log:
            raise ValueError(f"missing CPU-reference success: {name}")
    elif "PASS " not in log:
        raise ValueError(f"missing runner success: {name}")
    for path in list(directory.glob("core*.excp_log.dump")) + list(directory.glob("excp_log*.dump")):
        if path.stat().st_size:
            raise ValueError(f"nonempty exception dump: {path}")
    paths = sorted(directory.glob("core*.instr_popped_log.dump"))
    if not paths:
        raise ValueError(f"missing dispatch traces: {name}")
    cores = [audit(path, strips) for path in paths]
    active = [core for core in cores if core["active"]]
    if not active:
        raise ValueError(f"no active Vector core: {name}")
    if name.endswith("_wrap") and not any(c["mte2_vector_intersections"] for c in active):
        raise ValueError(f"no observed overlap: {name}")
    script_dir = Path(__file__).resolve().parent
    if name.startswith("aiv_") or name in ("regression_rows6", "regression_pages2", "regression_odd3"):
        rows = 6 if name == "regression_rows6" else 1
        command = [sys.executable, str(script_dir / "analyze_single_token_decode_trace.py"),
                   str(directory / "core0.veccore0.instr_popped_log.dump"),
                   str(directory / "core0.veccore0.instr_log.dump"),
                   "--rows", str(rows), "--strips", str(strips), "--items", "1"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (directory / "census.txt").write_text(result.stdout)
    else:
        command = [sys.executable, str(script_dir / "analyze_splitk_decode_trace.py"), str(directory), "--assert-batched"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (directory / "atomic.json").write_text(result.stdout)
    ticks = re.search(r"Total tick:\s*(\d+)", log)
    observations = [line.strip() for line in log.splitlines() if line.strip().startswith(
        ("guard bands:", "inputs:", "scores:", "numerics:", "egress:", "isolation:",
         "PASS Split-K boundary", "PASS: CPU references matched."))]
    summary = dict(status="PASS", case=name, strips_per_active_core=strips,
                   model_ticks=int(ticks.group(1)) if ticks else None,
                   observations=observations, idle_cores=len(cores) - len(active), cores=active)
    (directory / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    cases = BASELINE if args.baseline_only else BASELINE + REGRESSIONS
    results = [verify(args.directory, name, strips) for name, strips in cases]
    sources = {}
    repo = Path(__file__).resolve().parents[1]
    for filename in ("turboquant_single_token_decode_aiv.cpp", "turboquant_single_token_decode_splitk_aiv.cpp"):
        relative = Path("kernels/ops/turboquant") / filename
        current = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        built = hashlib.sha256((args.directory / "src" / relative).read_bytes()).hexdigest()
        if current != built:
            raise ValueError(f"workspace differs from tested source: {filename}")
        sources[str(relative)] = current
    manifest = dict(status="PASS", scope="baseline" if args.baseline_only else "baseline-and-regressions",
                    cases_passed=len(results), evidence_root=str(args.directory), source_sha256=sources,
                    results=results)
    (args.directory / "verified_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS {len(results)} cases: logs, exceptions, exact boundary gates, flags, census, wrap overlap, source hashes")


if __name__ == "__main__":
    main()
