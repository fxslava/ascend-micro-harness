#!/usr/bin/env python3
"""Discard only CANN preprocessing core objects consumed by its in-place link."""

from pathlib import Path
import sys


def prepare(build_dir):
    root = Path(build_dir).resolve(strict=True)
    removed = 0
    for name in ("aic_obj", "aiv_obj", "m200_obj"):
        for obj in (root / "CMakeFiles" / (name + ".dir")).rglob("*.o"):
            # Never follow a build-tree symlink into a source or another build.
            obj.resolve(strict=True).relative_to(root)
            with obj.open("rb") as stream:
                header = stream.read(18)
            if (len(header) == 18 and header[:4] == b"\x7fELF"
                    and header[5] == 1
                    and int.from_bytes(header[16:18], "little") == 2):
                obj.unlink()
                removed += 1
    return removed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: prepare_cann_preprocess.py PREPROCESS_BUILD_DIR")
    print("CANN preprocess: removed %d consumed core object(s) in %s"
          % (prepare(sys.argv[1]), sys.argv[1]))
