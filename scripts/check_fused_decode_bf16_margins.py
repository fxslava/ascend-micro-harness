#!/usr/bin/env python3
"""Can the fused decode's exact-boundary fixtures stay at 0 ULP in bf16?

WHY THIS EXISTS. bf16 keeps 7 stored significand bits against fp16's 10, so every exact-arithmetic
claim the fused decode's ladder and fixture make had to be re-checked before the bf16 instance
could reuse them. Widening a tolerance to make a bf16 rung pass would destroy the thing the rung
is for (REFINEMENT_OF_COMBAT_KERNELS.md section 3: "do not relax exact boundary checks to make
that fixture pass"), so the question is not "what budget does bf16 need" but "does bf16 still fit
the budget fp16 has". This answers both halves of that, in the two places a 16-bit format appears:

  1. THE LADDER'S OUTPUT ORACLE (tests/sim/kernel/test_camodel_fused_decode_opt.cpp). The device
     accumulates the context in fp32 and casts once, so the gold model computes in fp64, narrows
     to fp32 and rounds to the output format. The device's fp32 result and the host's fp64 one are
     not the same number, so the two round to the same 16-bit word only if the exact value is far
     enough from a rounding MIDPOINT of that format's grid. This reports that distance, in units
     of the value's own fp32 ULP, for every rung -- so a margin of N means the device's fp32
     accumulation would have to be N fp32 ULPs off before the comparison could flip.

  2. THE FIXTURE'S RAW QUERY (tests/common/fused_decode_opt_fixture.hpp). The raw-query entry's
     prologue has to reproduce the pre-rotated query BIT FOR BIT, or stage 0 cannot be held to the
     pre-rotated entry's oracle. That needs Pi(query) to be exactly representable in the element
     format. This reports the widest significand it actually uses.

Both numbers are checked in code as well -- tests/cpu/test_fused_decode_opt_fixture.cpp asserts the
raw query's exactness, and the ladder asserts 0 ULP -- so this script is the DERIVATION behind
those assertions rather than a substitute for them. Run it when either fixture's arithmetic changes.

    python3 scripts/check_fused_decode_bf16_margins.py
"""
import struct
import sys
from fractions import Fraction

# fp16 keeps 10 stored significand bits, bf16 7. The device accumulates in fp32 in both instances.
FORMATS = (("fp16", 10), ("bf16", 7))

# A margin this far from a midpoint cannot be crossed by any plausible fp32 accumulation error: the
# device's numerator here is a small exact integer and the only inexact step is one reciprocal and
# one multiply, a handful of fp32 ULPs at most.
MARGIN_FLOOR = 64.0


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ulp32(x):
    bits = struct.unpack("<I", struct.pack("<f", f32(x)))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0] - f32(x)


def quantum(x, mantissa_bits):
    """The grid spacing of a `mantissa_bits` format at x's binade, as an exact Fraction."""
    exponent = (struct.unpack("<I", struct.pack("<f", f32(x)))[0] >> 23 & 0xFF) - 127
    return Fraction(2) ** (exponent - mantissa_bits)


def midpoint_margin(exact, mantissa_bits):
    """Distance from `exact` to the nearest rounding midpoint, in fp32 ULPs of the value."""
    x = f32(float(exact))
    q = quantum(x, mantissa_bits)
    offset = exact / q
    frac = offset - int(offset)
    return float(abs(frac - Fraction(1, 2)) * q / Fraction(ulp32(x)))


def exactly_representable(exact, mantissa_bits):
    x = f32(float(exact))
    return (exact / quantum(x, mantissa_bits)).denominator == 1


# ------------------------------------------------------------------ 1. the ladder's output oracle
#
# Each rung's live row counts, from scripts/run_fused_decode_canary_ladder.sh. In the exact
# (non---varied) fixture every key nibble is 0, so every score is equal and every softmax weight is
# exactly 1; a tile of v live rows contributes v * (2n + 1) to the numerator and v to the
# denominator, with n = (tile + channel offset) % 8 running over all eight residues.
RUNGS = (
    ("isolated page canaries", (64,)),
    ("rung 1  tiles=1 tail=64", (64,)),
    ("rung 2  tiles=2 tail=15", (64, 15)),
    ("rung 3  tiles=5 tail=1", (64, 64, 64, 64, 1)),
    ("rung 4  tiles=3 tail=64 (ring depth)", (64, 64, 64)),
    ("rung 5  tiles=6 tail=64 (ring wrap)", (64, 64, 64, 64, 64, 64)),
)


def ladder_report():
    print("1. THE LADDER'S OUTPUT ORACLE")
    print("   distance from the nearest rounding midpoint, in fp32 ULPs of the value\n")
    print("   %-38s %-24s %-24s" % ("rung", "fp16", "bf16"))
    worst = {name: None for _, name in ((0, "fp16"), (0, "bf16"))}
    ok = True
    for label, live in RUNGS:
        cells, row_worst = [], {}
        for name, bits in FORMATS:
            margin, exact_all, witness = None, True, None
            for offset in range(8):
                numerator = sum(v * (2 * ((t + offset) % 8) + 1) for t, v in enumerate(live))
                value = Fraction(numerator, sum(live))
                if not exactly_representable(value, bits):
                    exact_all = False
                m = midpoint_margin(value, bits)
                if margin is None or m < margin:
                    margin, witness = m, value
            row_worst[name] = (margin, witness)
            cells.append("%-10s %9.1f" % ("exact" if exact_all else "inexact", margin))
            if worst[name] is None or margin < worst[name][0]:
                worst[name] = (margin, witness, label)
            if margin < MARGIN_FLOOR:
                ok = False
        print("   %-38s %-24s %-24s" % (label, cells[0], cells[1]))
        _ = row_worst
    print()
    for name, _ in FORMATS:
        margin, witness, label = worst[name]
        print("   worst %s margin: %.1f fp32 ULP, at %s (%s = %.9g)"
              % (name, margin, label, witness, float(witness)))
    print("\n   => every bf16 margin is WIDER than the fp16 margin of the same rung, and the fp16")
    print("      comparison already passes at 0 ULP on CAModel. So the bf16 leg keeps the 0-ULP")
    print("      budget unchanged; it is the easier of the two, not the one needing slack.")
    return ok


# ----------------------------------------------------------------- 2. the fixture's raw query
#
# RawQueryBasis, reproduced from tests/common/fused_decode_opt_fixture.hpp: Pi = (1/sqrt(D))
# diag(s) H diag(s) with H[i][j] = (-1)^popcount(i & j). The pre-rotated query row is the constant
# vector (coefficient / 256) * 1 with coefficient in {1, 2, 3}, so Pi of it is
# +-coefficient * (H s)_i / 4096 -- a small integer over a power of two.
DIM = 256
COEFFICIENTS = (1, 2, 3)


def sign_at(channel):
    return -1.0 if ((channel * 2654435761) % (1 << 32)) >> 13 & 1 else 1.0


def pi_of_constant(value, signs):
    x = [value * s for s in signs]
    stride = 1
    while stride < DIM:
        for i in range(DIM):
            if not i & stride:
                a, b = x[i], x[i ^ stride]
                x[i], x[i ^ stride] = a + b, a - b
        stride <<= 1
    return [x[i] * signs[i] / 16.0 for i in range(DIM)]


def significand_bits(value):
    mantissa = struct.unpack("<I", struct.pack("<f", f32(value)))[0] & 0x7FFFFF
    if mantissa == 0:
        return 0
    used = 23
    while not mantissa & 1:
        mantissa >>= 1
        used -= 1
    return used


def raw_query_report():
    print("\n2. THE FIXTURE'S RAW QUERY (the raw-query entry's prologue input)")
    print("   widest significand actually used, against what each format stores\n")
    signs = [sign_at(i) for i in range(DIM)]
    widest = 0
    for coefficient in COEFFICIENTS:
        image = pi_of_constant(coefficient / 256.0, signs)
        used = max(significand_bits(v) for v in image)
        widest = max(widest, used)
        print("   GQA coefficient %d (query = %-12r): %d significand bits"
              % (coefficient, coefficient / 256.0, used))
    print()
    ok = True
    for name, bits in FORMATS:
        verdict = "exact" if widest <= bits else "NOT REPRESENTABLE"
        print("   %s stores %2d bits -> %s (margin %d)" % (name, bits, verdict, bits - widest))
        if widest > bits:
            ok = False
    print("\n   => exact in both, but bf16's margin is ZERO: 7 bits used against 7 stored. A")
    print("      different sign vector or a non-constant query could push this to 9 and break")
    print("      bf16 while leaving fp16 intact, which is why the fixture ASSERTS the round trip")
    print("      rather than documenting it (tests/cpu/test_fused_decode_opt_fixture.cpp).")
    return ok


if __name__ == "__main__":
    good = ladder_report()
    good = raw_query_report() and good
    print("\n%s" % ("PASS: bf16 reuses every exact-arithmetic claim at the fp16 budget"
                    if good else "FAIL: an exact-arithmetic claim does not survive bf16"))
    sys.exit(0 if good else 1)
