#!/usr/bin/env python3
"""Audit MIX decode cross-core handoffs and once-per-worker query rotation.

Read *.instr_popped_log.dump, NOT *.instr_log.dump. The issue log drops
instructions: on the four-page Split-K rung AIV0's holds 2 of its 4
SET_CROSS_CORE and 2 of its 4 WAIT_FLAG_DEV -- instruction IDs 035718, 042737,
074351 and 081403 are absent from it and present in the popped log, although
both run to ID ~157830. A census over the issue log reports a correct kernel as
imbalanced and, worse, would pass an incorrect one vacuously.

With --workers/--pages/--strips, also
check 64 rotation Mmads per active worker plus 8 score Mmads per processed
strip, and that each cross-core channel carried exactly one announcement per
strip. Counts can span multiple launches.

Both inter-core synchronization modes are read, because they are different
instructions AND different pairing rules.

  mode 4  `set_intra_block` / `wait_intra_block`, printed as SET_INTRA_BLOCK /
          WAIT_INTRA_BLOCK with a `sync_id`. One AIC and ONE AIV, so a channel
          balances when that pair's sets equal its waits. AIV1 reports
          sub-block-local ids; rebase by 16 to the id the AIC addresses.

  mode 2  `ffts_cross_core_sync` / `wait_flag_dev`, printed as SET_CROSS_CORE /
          WAIT_FLAG_DEV with the id in an XT operand (packed into GetffstMsg's
          `1 + (mode << 4) + (flagId << 8)` on the set). The AIC and BOTH AIVs,
          with the same id on every core, so it does NOT balance pairwise:
          a quorum is one announcement from EACH AIV against one AIC wait, and
          a broadcast is one AIC announcement against one wait on EACH AIV.
          Checking mode 2 with mode 4's rule reports a correct kernel as
          imbalanced; checking only mode 4 passes a mode-2 kernel vacuously.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

LINE = re.compile(r"\[\d+\].*?\(ID: \d+\)\s+(\w+)(.*)")
# `FLAG ID:0` prints without the 0x the other ids carry, so the prefix has to be
# optional -- it is not cosmetic, and a required `0x` drops every place on event
# id 0, which at one strip per page is the only slot doing real work.
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(?:0x)?([0-9a-f]+)")
MODE4 = re.compile(r"sync_id:(\d+)")
XT_OPERAND = re.compile(r"XT:X\d+=(0x[0-9a-fA-F]+|\d+)")
FFTS_MODE_SHIFT = 4
FFTS_FLAG_SHIFT = 8

ROLES = (("cubecore0", "AIC"), ("veccore0", "AIV0"), ("veccore1", "AIV1"))


def role_of(name):
    for marker, role in ROLES:
        if ".%s." % marker in name:
            return role
    return "?"


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--strips", type=int)
    args = parser.parse_args()
    sets, waits, ops = Counter(), Counter(), Counter()
    # mode 4 keeps its pairwise census; mode 2 is counted per (channel, role).
    m4_sets, m4_waits = Counter(), Counter()
    m2 = defaultdict(Counter)
    for path in args.directory.glob("core*.instr_popped_log.dump"):
        core = int(re.match(r"core(\d+)", path.name)[1])
        role = role_of(path.name)
        rebase = 16 if role == "AIV1" else 0
        with path.open(errors="replace") as source:
            for raw in source:
                match = LINE.search(raw)
                if not match:
                    continue
                op, tail = match.groups()
                ops[op] += 1
                if op in ("SET_FLAG", "WAIT_FLAG"):
                    flag = FLAG.search(tail)
                    if flag:
                        pipe, target, number = flag.groups()
                        number = int(number, 16)
                        if number >= 6:
                            raise ValueError(f"{path.name}: reserved HardEvent ID {number}")
                        key = path.name, pipe, target, number
                        (sets if op == "SET_FLAG" else waits)[key] += 1
                elif op in ("SET_INTRA_BLOCK", "WAIT_INTRA_BLOCK"):
                    flag = MODE4.search(tail)
                    if flag:
                        key = core, int(flag[1]) + rebase
                        (m4_sets if op == "SET_INTRA_BLOCK" else m4_waits)[key] += 1
                elif op in ("SET_CROSS_CORE", "WAIT_FLAG_DEV"):
                    found = XT_OPERAND.search(tail)
                    if not found:
                        continue
                    value = int(found[1], 0)
                    if op == "SET_CROSS_CORE":
                        if value & 0x1 != 0x1:
                            continue  # not a GetffstMsg payload
                        mode = (value >> FFTS_MODE_SHIFT) & 0x3
                        number = (value >> FFTS_FLAG_SHIFT) & 0xF
                        m2["m%d:%d" % (mode, number)]["set." + role] += 1
                    else:
                        # The wait carries the raw id. No rebase: a mode-2 id
                        # names the same channel on every core in the block.
                        m2["m2:%d" % (value & 0xF)]["wait." + role] += 1

    if not sets or sets != waits:
        raise ValueError(f"HardEvent imbalance: {sets - waits}, {waits - sets}")
    if m4_sets != m4_waits:
        raise ValueError(f"mode-4 imbalance: {m4_sets - m4_waits}, {m4_waits - m4_sets}")
    if not m4_sets and not m2:
        raise ValueError("no cross-core handshake of either mode in this trace")

    mode4 = Counter()
    for (_, flag), count in m4_sets.items():
        mode4[flag] += count

    # Mode-2 pairing, per channel and per direction.
    for channel, census in sorted(m2.items()):
        quorum = (census["set.AIV0"], census["set.AIV1"], census["wait.AIC"])
        broadcast = (census["set.AIC"], census["wait.AIV0"], census["wait.AIV1"])
        if any(quorum) and len(set(quorum)) != 1:
            raise ValueError(
                f"{channel} quorum imbalance: AIV0 announces {quorum[0]}, AIV1 "
                f"announces {quorum[1]}, the AIC waits {quorum[2]}")
        if any(broadcast) and len(set(broadcast)) != 1:
            raise ValueError(
                f"{channel} broadcast imbalance: the AIC announces {broadcast[0]}, "
                f"AIV0 waits {broadcast[1]}, AIV1 waits {broadcast[2]}")

    mmads = sum(count for op, count in ops.items() if "MMAD" in op)
    if any(v is not None for v in (args.workers, args.pages, args.strips)):
        if any(v is None for v in (args.workers, args.pages, args.strips)):
            parser.error("supply --workers, --pages and --strips together")
        strips = args.pages * args.strips
        expected = args.workers * 64 + strips * 8
        if mmads != expected:
            raise ValueError(f"Mmad census mismatch: {mmads}/{expected}")
        # One announcement per strip on each channel, in each direction it is
        # used. The continuous strip pipeline rendezvouses per STRIP, not per
        # page, so the page count only enters through the strip total.
        carried = {}
        for channel, census in m2.items():
            if census["set.AIV0"]:
                carried[channel + " readiness"] = census["set.AIV0"]
            if census["set.AIC"]:
                carried[channel + " release"] = census["set.AIC"]
        if not carried:
            raise ValueError("no mode-2 channel carried any announcement; the "
                             "strip census has nothing to check")
        wrong = {name: n for name, n in carried.items() if n != strips}
        if wrong:
            raise ValueError(f"strip census mismatch: expected {strips} "
                             f"announcements per channel, got {wrong}")

    print(json.dumps(dict(hardevent_pairs=sum(sets.values()), reserved_event_ids=0,
                          mode4_sets_and_waits=dict(mode4),
                          mode2_channels={k: dict(v) for k, v in sorted(m2.items())},
                          cube_mmads=mmads), indent=2))


if __name__ == "__main__":
    main()
