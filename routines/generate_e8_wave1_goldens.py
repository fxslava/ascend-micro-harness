#!/usr/bin/env python3
"""Verify or reproduce the frozen Wave 1 BF16/INT8 E8 golden payloads.

Regeneration uses the byte-identical original oracle in a temporary native
workspace, so its diagnostic report never writes beside the checked-out source.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(folder, records):
    for item in records:
        name = item["file"]
        if Path(name).name != name:
            raise ValueError("fixture name must be a basename")
        path = folder / name
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("payload mismatch: " + str(path))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--output", type=Path, help="new or empty native WSL output directory")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "e8_wave1_export.json").read_text())
    fixture = ROOT / "e8_wave1_goldens"
    if sha(fixture / "manifest.json") != provenance["goldens"]["manifest_sha256"]:
        raise ValueError("frozen fixture manifest changed")
    records = json.loads((fixture / "manifest.json").read_text())["files"]
    verify(fixture, records)
    for item in provenance["headers"] + [provenance["canonical_generator"]]:
        if sha(ROOT / item["file"]) != item["sha256"]:
            raise ValueError("frozen source mismatch: " + item["file"])
    result = dict(status="PASS_FROZEN_BYTES", payloads=len(records), elements_per_dtype=8192)
    if args.output is not None:
        import numpy as np
        if np.__version__ != "2.2.6":
            raise SystemExit("Exact regeneration requires numpy==2.2.6")
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise SystemExit("--output must be new or empty")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="e8-wave1-", dir=output.parent) as temp:
            stage = Path(temp)
            (stage / "scripts").mkdir()
            oracle = stage / "scripts/e8_reference.py"
            shutil.copyfile(ROOT / "e8_reference.py", oracle)
            run = subprocess.run([sys.executable, "-B", str(oracle), "--fixture-output",
                                  str(output), "--stress-fixture"], check=True,
                                 stdout=subprocess.PIPE, text=True)
            report = json.loads(run.stdout)
            verify(output, records)
            generated = json.loads((output / "manifest.json").read_text())["files"]
            if generated != records:
                raise ValueError("regenerated payload manifest differs")
            shutil.copyfile(stage / "reports/e8_reference_validation.json",
                            output / "reference_validation.json")
            result.update(status="PASS_EXACT_REGENERATION", numpy=np.__version__,
                          oracle_status=report["status"], output=str(output))
        (output / "reproduction_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
