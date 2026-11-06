"""Snapshot the upstream decode dependencies; subsequent harness edits are local.

Run only for the initial import, since this intentionally refuses to overwrite.
"""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Ascend\vllm-ascend")
upstream = source / "csrc/attention/turboquant/op_kernel"
dest = root / "kernels/ops/turboquant/fused_decode"
files = ["common/turboquant_common.h", "common/turboquant_layout.h",
         "common/turboquant_mode.h", "common/turboquant_codec_950.h",
         "common/turboquant_codec_mx.h", "cube/turboquant_cube_service.h",
         "cube/turboquant_query_basis.h", "vector/turboquant_vector_service.h"]
mapping = [(upstream / f, dest / f) for f in files]
# The optimizations live once, in the subtree exported to vllm-ascend, under the name that
# integration expects. Upstream still ships the file as turboquant_fused_decode_opt.cpp.
mapping += [(upstream / "turboquant_fused_decode_opt.cpp",
             dest.parent / "vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp"),
            (source / "csrc/kernels/types.h", dest / "types.h")]
if any(dst.exists() for _, dst in mapping):
    raise SystemExit("Import destinations already exist; refusing to overwrite local work")
manifest = {}
for src, dst in mapping:
    data = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    manifest[str(src.relative_to(source)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
(dest / "upstream_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Imported {len(mapping)} files")
