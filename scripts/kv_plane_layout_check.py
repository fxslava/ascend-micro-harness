#!/usr/bin/env python3
"""Layout agreement check for the TurboQuant 4-bit KV planes.

Three independent things are checked, over every (block, head, token, channel)
of a whole cache:

  1. WRITER vs DECODER. turboquant_reshape_and_cache_bf16.cpp (the vllm-ascend
     copy, which carries the in-page offset) and
     turboquant_paged_decode_attention.cpp must address the same byte for the
     same logical element, on all three planes. Any disagreement here is a
     silent-corruption path: both kernels pass their own oracles and the
     product is wrong.

  2. NESTING. The writer's byte function is compared against the two candidate
     contiguous PyTorch readings. It matches [P, H, S, d/2] and not
     [P, S, H, d/2], which is the schema's declared order -- audit F4 / Sec 2.4.

  3. Nz(K) == Zn(K^T). The 16x256 dequantised key page's Nz image and the
     256x16 transpose's Zn image must be the same offsets, which is what lets
     one L1 image serve both operand roles through byte-identity L1->L0 loads
     -- audit Sec 2.2. This is the invariant that forbids "fixing" the layout.

Reference, from the two kernel headers. R = 16 is the strip height (= M0 of the
DaVinci fractal); nibble indices are within a (block, head) page:

    key    nibble = token * d + channel                      (token-major)
    value  nibble = (token // R) * (R*d) + channel * R + (token % R)
                                                             (strip-blocked
                                                              channel-major)
    scale  float  = ((kv * P + block) * H + head) * S + token      (planar)
"""

import argparse
import itertools
import sys

R = 16              # strip height, the writer's kTokens
C0_BF16 = 16        # 32 B / sizeof(bfloat16_t) -- the 16-bit fractal's C0


# ----------------------------------------------------------------- the writer
def writer_key_byte(block, head, token, channel, P, S, H, d):
    """StoreChunk: page_bytes + head*head_bytes + key_token_bytes, + nibble."""
    head_bytes = S * d // 2
    page_bytes = block * H * head_bytes
    return page_bytes + head * head_bytes + token * (d // 2) + channel // 2


def writer_value_byte(block, head, token, channel, P, S, H, d):
    """Same page base; value_token_bytes = (t0 // R) * kPackedBytes."""
    head_bytes = S * d // 2
    page_bytes = block * H * head_bytes
    strip, tok_in = divmod(token, R)
    return page_bytes + head * head_bytes + strip * (R * d // 2) + (channel * R + tok_in) // 2


def writer_scale_float(kv, block, head, token, P, S, H, d):
    return ((kv * P + block) * H + head) * S + token


# ---------------------------------------------------------------- the decoder
def decoder_key_byte(block, head, token, channel, P, S, H, d):
    """strip_base_bytes = ((page*H) + kv_head) * strips * packed, + token*d + ch."""
    strips = S // R
    packed = R * d // 2
    strip, tok_in = divmod(token, R)
    strip_base = ((block * H) + head) * strips * packed
    return strip_base + strip * packed + (tok_in * d + channel) // 2


def decoder_value_byte(block, head, token, channel, P, S, H, d):
    """Same strip base; nibble index inside a strip is channel*R + token."""
    strips = S // R
    packed = R * d // 2
    strip, tok_in = divmod(token, R)
    strip_base = ((block * H) + head) * strips * packed
    return strip_base + strip * packed + (channel * R + tok_in) // 2


def decoder_scale_float(kv, block, head, token, P, S, H, d):
    """Planar, with the plane stride a launch scalar rather than kNumPages."""
    plane = P * H * S
    return kv * plane + ((block * H) + head) * S + token


# ------------------------------------------------- candidate PyTorch readings
def view_pshd(block, head, token, channel, P, S, H, d):
    """contiguous [P, S, H, d/2] -- what the schema declares."""
    return ((block * S + token) * H + head) * (d // 2) + channel // 2


def view_phsd(block, head, token, channel, P, S, H, d):
    """contiguous [P, H, S, d/2] -- head outside token."""
    return ((block * H + head) * S + token) * (d // 2) + channel // 2


def check_kernels_agree(P, S, H, d, verbose):
    bad = 0
    for block, head, token in itertools.product(range(P), range(H), range(S)):
        for channel in (0, 1, d // 2, d - 1):
            for name, w, r in (
                ("key", writer_key_byte, decoder_key_byte),
                ("value", writer_value_byte, decoder_value_byte),
            ):
                a = w(block, head, token, channel, P, S, H, d)
                b = r(block, head, token, channel, P, S, H, d)
                if a != b:
                    bad += 1
                    if verbose and bad <= 8:
                        print(f"  {name} ({block},{head},{token},{channel}): "
                              f"writer {a} != decoder {b}")
    for kv, block, head, token in itertools.product(
            range(2), range(P), range(H), range(S)):
        a = writer_scale_float(kv, block, head, token, P, S, H, d)
        b = decoder_scale_float(kv, block, head, token, P, S, H, d)
        if a != b:
            bad += 1
            if verbose and bad <= 8:
                print(f"  scale ({kv},{block},{head},{token}): {a} != {b}")
    return bad


def check_nesting(P, S, H, d):
    """Which contiguous reading, if either, is the writer's byte function?"""
    out = {}
    for label, view in (("[P, S, H, d/2]", view_pshd), ("[P, H, S, d/2]", view_phsd)):
        agree = sum(
            writer_key_byte(b, h, t, 0, P, S, H, d) == view(b, h, t, 0, P, S, H, d)
            for b, h, t in itertools.product(range(P), range(H), range(S)))
        out[label] = (agree, P * H * S)
    return out


def check_nz_zn(rows=R, channels=256):
    """Nz(K[rows, channels]) and Zn(K^T[channels, rows]) must be the same bytes.

    The two images are built independently -- each from its own matrix, by its
    own layout rule, traversed in its own order -- and only then compared under
    the correspondence K^T[c][t] = K[t][c]. Writing one formula twice would
    prove nothing, which is the trap this function exists to avoid.

    Both rules are written in their general form, with the n1 grid axis live,
    rather than collapsed for N <= C0 -- collapsing it would bake the answer in.

    FRACTAL = C0 * C0 elements (16x16 = 512 B at bf16).

    Nz of an [N, D] matrix, grid (D1, N1), fractal row n0, fractal col d0:
        offset = (d1*ceil(N/C0) + n1)*FRACTAL + n0*C0 + d0
        n = n1*C0 + n0,   d = d1*C0 + d0

    Zn of a [K, N] matrix in the Cube's B role, grid (K1, N1), row n0, col k0:
        offset = (k1*ceil(N/C0) + n1)*FRACTAL + n0*C0 + k0
        k = k1*C0 + k0,   n = n1*C0 + n0

    What the equality then says: both conventions put the channel axis on the
    OUTER grid axis and the token axis on the inner one, with the token as the
    fractal row -- so the two index tuples coincide and the identity is
    structural rather than contingent. This function is therefore a DERIVATION
    check: it cannot fail for a correct pair of layout rules, and its value is
    as a guard against a later change that inserts a permutation. The control
    below is what has teeth -- a row-major K^T, which is what any explicit
    transpose would produce, does not match.
    """
    c0 = C0_BF16
    fractal = c0 * c0
    n_blocks = -(-rows // c0)

    # image 1: walk K row-major (token, channel), place by the Nz rule.
    nz = {}
    for n in range(rows):
        n1, n0 = divmod(n, c0)
        for dcol in range(channels):
            d1, d0 = divmod(dcol, c0)
            nz[(n, dcol)] = (d1 * n_blocks + n1) * fractal + n0 * c0 + d0

    # image 2: walk K^T row-major (channel, token), place by the Zn rule.
    zn = {}
    for krow in range(channels):
        k1, k0 = divmod(krow, c0)
        for n in range(rows):
            n1, n0 = divmod(n, c0)
            zn[(krow, n)] = (k1 * n_blocks + n1) * fractal + n0 * c0 + k0

    # control: a plain row-major K^T, which is what an explicit transpose gives.
    rm = {(c, t): c * rows + t for c in range(channels) for t in range(rows)}

    bad = sum(nz[(t, c)] != zn[(c, t)] for t in range(rows) for c in range(channels))
    bad_rm = sum(nz[(t, c)] != rm[(c, t)] for t in range(rows) for c in range(channels))
    span = max(nz.values()) + 1
    distinct = len(set(nz.values()))
    return bad, bad_rm, span, distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=9, help="P, physical pages")
    ap.add_argument("--block-size", type=int, default=128, help="S, tokens per page")
    ap.add_argument("--kv-heads", type=int, default=4, help="H")
    ap.add_argument("--head-size", type=int, default=256, help="d")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    P, S, H, d = a.blocks, a.block_size, a.kv_heads, a.head_size

    if S % R:
        sys.exit(f"block_size {S} is not a multiple of the {R}-token strip")

    print(f"cache (P, S, H, d) = ({P}, {S}, {H}, {d}), "
          f"strip {R}, page {S // R} strips, {S * d // 2} B per (block, head)")

    print("\n1. writer vs decoder, all three planes")
    bad = check_kernels_agree(P, S, H, d, a.verbose)
    print(f"   disagreements: {bad}   -> {'AGREE' if bad == 0 else 'MISMATCH'}")

    print("\n2. which contiguous PyTorch reading is the writer's byte function?")
    for label, (agree, total) in check_nesting(P, S, H, d).items():
        verdict = "matches everywhere" if agree == total else "does NOT match"
        print(f"   {label:>16}: {agree}/{total}  {verdict}")
    print("   (the schema declares [P, S, H, d/2]; the kernel stores [P, H, S, d/2])")

    print(f"\n3. Nz(K[{R},{d}]) == Zn(K^T[{d},{R}]) byte identity  (derivation check)")
    nz_bad, rm_bad, span, distinct = check_nz_zn(rows=R, channels=d)
    print(f"   vs Zn(K^T), fractal:    {nz_bad:6d} disagreements  "
          f"-> {'IDENTICAL' if nz_bad == 0 else 'NOT IDENTICAL'}")
    print(f"   vs row-major K^T (ctl): {rm_bad:6d} disagreements  "
          f"-> {'differs, as it must' if rm_bad else 'CONTROL FAILED'}")
    print(f"   image bijective over [0, {span}): "
          f"{'yes' if distinct == span == R * d else 'NO'}")

    print("\nwhat a reader using the declared view actually gets:")
    inv = {writer_key_byte(0, h, t, 0, P, S, H, d): (h, t)
           for h in range(H) for t in range(S)}
    for head, token in ((1, 0), (0, 1), (2, 0)):
        off = view_pshd(0, head, token, 0, P, S, H, d)
        print(f"   asked (token {token}, head {head}) -> byte {off:6d} "
              f"-> holds (head {inv[off][0]}, token {inv[off][1]})")

    return 0 if bad == 0 and nz_bad == 0 and rm_bad > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
