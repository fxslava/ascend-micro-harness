#!/usr/bin/env python3
"""UB arena budget for turboquant_paged_decode_attention as a function of query rows.

Mirrors the kernel's own allocator exactly: 32-byte operand bases, one 32-byte
DataBlock of bank skew between live slots, and a VECTOR_REG_WIDTH (256 B) read
pad behind the last operand (rule AKA3003). The slot order is the `NextSlot`
chain in kernels/ops/turboquant/turboquant_paged_decode_attention.cpp, so a
change there must be mirrored here or the table stops meaning anything.

`--probe` keeps the two verification-only buffers the probe carries (the
per-strip score archive and the pre-gate `out_rotated` panel); production drops
them. Reproduces the table in reports/qwen35_turboquant_regime_audit.md Sec 3.3.
"""

import argparse

BLOCK = 32          # DataBlock / minimum operand alignment
SKEW = 32           # one DataBlock of bank skew between live slots
READ_PAD = 256      # VECTOR_REG_WIDTH on dav-3510
F32, B16, I16 = 4, 2, 2

HEAD_SIZE = 256     # d
KEYS = 16           # keys per strip == the strip height
PANEL = KEYS * HEAD_SIZE
PACKED = PANEL // 2
STATE_LANES = 8     # one fp32 DataBlock, the replicated-state invariant
BC_LANES = 64       # one Brcb repeat, 256 B
DEPTH = 2           # ring depth on the UB landings and the score windows
STRIPS = 8          # strips per 128-token page
GATHER = 3          # sub-DataBlock egress stages


def _round_up(b):
    return (b + BLOCK - 1) // BLOCK * BLOCK


def _next(base, size):
    return _round_up(base + size) + SKEW


def arena_bytes(rows, probe):
    """Byte offset just past the last operand, plus the read pad."""
    ring_score = rows * KEYS * F32 + SKEW
    ring_packed = PACKED + SKEW
    ring_key = PANEL * B16 + SKEW
    ring_value = PANEL * F32 + SKEW
    ring_scale = 2 * KEYS * F32 + SKEW

    o = 0
    for size in (
        DEPTH * ring_score,     # Fixpipe landing ring, pinned at UB 0
        DEPTH * ring_packed,    # packed key strips
        DEPTH * ring_packed,    # packed value strips
        DEPTH * ring_key,       # dequantised bf16 key pages
        DEPTH * ring_value,     # dequantised fp32 value pages
        DEPTH * ring_scale,     # key + value scale pairs
        PANEL * I16,            # signed dequant scratch, single buffered
        rows * STATE_LANES * F32,   # m_prev
        rows * STATE_LANES * F32,   # l_prev
        rows * HEAD_SIZE * F32,     # acc
    ):
        o = _next(o, size)

    for _ in range(7):          # m_new l_new alpha alpha_delta l_scaled tile_max p_sum
        o = _next(o, STATE_LANES * F32)
    for _ in range(4):          # tile_max_bc m_bc p_sum_bc alpha_bc
        o = _next(o, BC_LANES * F32)
    for _ in range(4):          # scaled_row centered p p_scaled
        o = _next(o, KEYS * F32)
    o = _next(o, BC_LANES * F32)        # p_x4
    o = _next(o, PANEL * F32)           # weighted
    o = _next(o, HEAD_SIZE * F32)       # weighted_sum
    o = _next(o, HEAD_SIZE * F32)       # acc_scaled

    if probe:
        o = _next(o, STRIPS * ring_score)   # score archive, verification only

    for _ in range(8):          # l_bc egress_a egress_b permuted signed_tmp denom gate pi
        o = _next(o, HEAD_SIZE * F32)
    o = _next(o, GATHER * HEAD_SIZE * F32)  # gather offsets
    o = _next(o, GATHER * HEAD_SIZE * F32)  # gather signs
    o = _next(o, rows * HEAD_SIZE * B16)    # context

    end = o + (rows * HEAD_SIZE * F32 if probe else 0)   # out_rotated, verification only
    return end + READ_PAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=6,
                    help="GQA group ratio H_q / H_kv (6 for Qwen3.5-27B)")
    ap.add_argument("--window", type=int, default=192,
                    help="the kernel's self-imposed kUbWindow, KiB")
    ap.add_argument("--ub", type=int, default=256,
                    help="physical AIV UB, KiB (prefer GetUBSizeInBytes() in code)")
    ap.add_argument("--rows", type=int, nargs="*",
                    default=[6, 12, 16, 18, 24, 32, 48, 64, 96, 128])
    args = ap.parse_args()

    print(f"{'rows':>5} {'B x (1+k)':>10} {'fractals':>9} "
          f"{'probe':>11} {'production':>12}  {args.window}KiB  {args.ub}KiB")
    for rows in args.rows:
        probe = arena_bytes(rows, True)
        prod = arena_bytes(rows, False)
        print(f"{rows:5d} {rows / args.group:10.2f} {-(-rows // 16):9d} "
              f"{probe / 1024:9.2f}KiB {prod / 1024:10.2f}KiB  "
              f"{'yes' if prod <= args.window * 1024 else 'NO ':>6}  "
              f"{'yes' if prod <= args.ub * 1024 else 'NO':>6}")

    built = arena_bytes(16, True)
    print(f"\nas built (rows=16, probe): {built} B = {built / 1024:.2f} KiB")
    print(f"freed by dropping the two verification buffers at rows=16: "
          f"{(built - arena_bytes(16, False)) / 1024:.2f} KiB")


if __name__ == "__main__":
    main()
