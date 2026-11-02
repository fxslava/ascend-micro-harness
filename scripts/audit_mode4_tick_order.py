#!/usr/bin/env python3
"""Audit CAModel traces for capacity-one ORDER and for cross-core STALL.

`analyze_aic_splitk_trace.py` counts SETs and WAITs per place and fails on an
imbalance. That check cannot see either fault class
`REFINEMENT_OF_COMBAT_KERNELS.md` SS1 and SS6.2 are about. A second SET issued
before the first was consumed balances every count and still loses a token. A
WAIT that stops blocking balances every count and still lets its consumer read
a generation early. Both are order properties of the completion log, so they
need the tick column, and the second one needs a little more than that.

Two checks, and the second is the one that found the four-page Split-K defect.

1. Intra-core HardEvent places, keyed by the operand ROUTE (pipe -> trigger
   pipe, id) and not by the queue column, must alternate SET, WAIT, SET,
   WAIT ... A `double SET` is a lost token; a `WAIT before SET` is an early
   read; a leftover token is an undrained place.

2. Cross-core mode-4 channels get a STALL census instead of a tick order,
   because of how the log reports a wait. All three of a block's
   `*.instr_popped_log.dump` files share one tick column -- on the four-page
   K=1 Split-K run the Cube's log and its `ccu_log.dump` both end at 88339 --
   but a `WAIT_INTRA_BLOCK` is logged at the tick it was ISSUED, not the tick
   it was released. So "wait before set" is the normal reading of a wait that
   blocked, and comparing the two tick columns directly reports an artefact:
   on that run AIV0's wait on `kFlagScoreReady` is logged before the Cube's
   set every page, yet the panel it then reads is bit-exact in all 16 rows.

   What the release tick CAN be read from is the scalar drain a correctly
   fenced kernel places right behind the wait. A cross-core wait blocks only
   the issuance of its own pipe (`CrossCoreWaitFlag` on Ascend 950PR), so the
   `SetFlag<x_S>` / `WaitFlag<x_S>` pair behind it stalls instruction issue
   until pipe x has retired past the wait -- and that `WaitFlag` is a SCALAR
   instruction whose logged tick IS when it ran. The gap from the wait to that
   `WaitFlag` is therefore the stall the wait actually imposed.

   It has to be that `WaitFlag` and not merely the next SCALAR instruction.
   Whatever the compiler schedules in between -- address arithmetic for the
   next call, a register spill -- is issued while the wait is still blocking,
   so measuring to it reports the 1-tick floor for a channel that is perfectly
   well fenced. All three of the decode operator's channels read [1, 1, 1, 1]
   under the looser rule and were marked `no-fence` although each one carries
   its drain.

   This is worth a gate because a wait that silently stops blocking looks
   exactly like a wait that had nothing to wait for. On the four-page rung
   AIV1's four waits on `kFlagPageDrainedV1`, a
   `CrossCoreWaitFlag<4, PIPE_MTE3>`, stalled 23146, 3, 3 and 3 ticks while
   the Cube's four announcements of that id are ~20600 ticks apart.
   `--expect-stall N` fails a channel whose later waits collapse like that,
   which is the regression test for it.

   The stall threshold is a proxy, though, and only a good one when the
   producer runs unbounded. Once a pipeline is bounded by a credit ring, a wait
   that finds its token already banked legitimately stalls only the cost of its
   own drain -- around a hundred ticks -- and no fixed threshold separates that
   from a latch that stopped blocking. So the release tick is also checked
   DIRECTLY against the announcement that should have produced it: a consuming
   counter releases the j-th wait with the j-th announcement, so release[j] must
   not precede announcement[j]. For a mode-2 quorum, where both AIVs announce,
   announcement[j] is the LATER of their j-th announcements. That check needs no
   threshold and is reported as `early retirement`.

   **A channel with no scalar drain behind its wait reports a FLOOR, not a
   stall**, and the census marks it `no-fence`. The Cube's `kFlagKeyReady`
   wait reports [2, 2, 2, 2] on the same run and is perfectly well ordered:
   the wait gates MTE1 while the scalar stream runs on, so there is nothing
   for this metric to read. Such a channel is never flagged, and proving IT
   ordered anything needs a different measurement -- for AIV0's
   `kFlagScoreReady` wait on PIPE_V, which also reports [1, 1, 1, 1] here,
   what established it was that AIV0's four recurrence spans (19386, 19386,
   19385, 19385 ticks) each START at the Cube's matching set (6929, 27600,
   48125, 68661). Read the floor as "not measured".

Both inter-core synchronization modes are read, because they are different
instructions. Mode 4 lowers to set_intra_block / wait_intra_block and prints as
`SET_INTRA_BLOCK` / `WAIT_INTRA_BLOCK sync_id:N`; mode 2 lowers to
ffts_cross_core_sync / wait_flag_dev and prints as `SET_CROSS_CORE` /
`WAIT_FLAG_DEV`, with the id in an `XT:Xn=0x..` operand and, on the set, packed
into GetffstMsg's `1 + (mode << 4) + (flagId << 8)`. Channels are named
`m4:<id>` and `m2:<id>` and never pooled: the two id spaces are independent on
Ascend 950PR, so id 8 in one is not id 8 in the other.

usage: audit_mode4_tick_order.py EVIDENCE_DIR [--block N] [--expect-stall N]
                                 [--cross] [--json]
       audit_mode4_tick_order.py --self-test
exit 0 clean, 1 on a violation, 2 on no usable trace.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TICK = re.compile(r"^\[\w+\]\s+\[(\d+)\]")
ENGINE = re.compile(r"\)\s+(\w+)\s*:")
OP = re.compile(r"\(ID: \d+\)\s+(\w+)")
# `FLAG ID:0` prints without the 0x the other ids carry, so the prefix has to
# be optional. It is not cosmetic: at one strip per page, slot 0 is the only
# slot doing real work, and a required `0x` drops every one of its places.
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(?:0x)?([0-9a-f]+)")
MODE4 = re.compile(r"sync_id:(\d+)")
# Mode 2. The wait carries the raw flag id; the set carries GetffstMsg's packed
# message, `1 + ((mode & 3) << 4) + ((flagId & 0xf) << 8)`, which is how a
# SET_CROSS_CORE of id 9 in mode 2 prints as 0x921.
XT_OPERAND = re.compile(r"XT:X\d+=(0x[0-9a-fA-F]+|\d+)")
FFTS_MODE_SHIFT = 4
FFTS_FLAG_SHIFT = 8


def ffts_message(raw):
    """-> (mode, flagId) for a SET_CROSS_CORE line, or None."""
    found = XT_OPERAND.search(raw)
    if not found:
        return None
    value = int(found[1], 0)
    if value & 0x1 != 0x1:
        return None  # not a GetffstMsg payload
    return ((value >> FFTS_MODE_SHIFT) & 0x3, (value >> FFTS_FLAG_SHIFT) & 0xF)
CORE = re.compile(r"core(\d+)\.(\w+)\.instr_popped_log\.dump")


def read(directory, block):
    """-> (places, waits, files).

    places: {place: [(tick, kind)]} for the intra-core HardEvent routes.
    waits:  [{core, engine, id, tick, released, stall}] per cross-core wait,
            where `released` is the tick of the SCALAR WAIT_FLAG behind it.
    sets:   {(channel, announcing engine): [tick]}, kept per ANNOUNCER so that
            a mode-2 quorum's j-th increment can be taken as the later of its
            two producers' j-th announcements.
    """
    places, waits, sets = defaultdict(list), [], defaultdict(list)
    seen = 0
    for path in sorted(directory.glob("core*.instr_popped_log.dump")):
        name = CORE.match(path.name)
        if not name:
            continue
        core, engine = int(name[1]), name[2]
        if block is not None and core != block:
            continue
        # AIV1 reports sub-block-local mode-4 ids; rebase to the absolute id
        # the AIC addresses, as the existing census does.
        rebase = 16 if engine.startswith("veccore1") else 0
        seen += 1
        pending = []
        with path.open(errors="replace") as source:
            for raw in source:
                tick, op = TICK.match(raw), OP.search(raw)
                if not tick or not op:
                    continue
                kind, tick = op[1], int(tick[1])
                unit = ENGINE.search(raw)
                unit = unit[1] if unit else "?"
                # A blocked wait shows up as a gap before the SCALAR half
                # of the x_S drain behind it -- a WAIT_FLAG issued on the
                # scalar unit. Not merely the next SCALAR instruction: see the
                # module docstring.
                if pending and unit == "SCALAR" and kind == "WAIT_FLAG":
                    for entry in pending:
                        entry["released"] = tick
                        entry["stall"] = tick - entry["tick"]
                    pending = []
                if kind in ("SET_INTRA_BLOCK", "WAIT_INTRA_BLOCK"):
                    found = MODE4.search(raw)
                    if not found:
                        continue
                    flag = "m4:%d" % (int(found[1]) + rebase)
                    if kind == "SET_INTRA_BLOCK":
                        sets[(flag, engine)].append(tick)
                    else:
                        entry = {"core": core, "engine": engine, "id": flag,
                                 "pipe": (re.search(r"PIPE:(\w+)", raw) or [None, "?"])[1],
                                 "tick": tick, "released": None, "stall": None}
                        waits.append(entry)
                        pending.append(entry)
                elif kind in ("SET_CROSS_CORE", "WAIT_FLAG_DEV"):
                    # Mode 2 (and modes 0/1) go out over FFTS instead. No +16
                    # rebase here: a mode-2 id names the same channel on the
                    # AIC and on both AIVs, which is what makes it a quorum.
                    if kind == "SET_CROSS_CORE":
                        message = ffts_message(raw)
                        if message is None:
                            continue
                        mode, number = message
                        sets[("m%d:%d" % (mode, number), engine)].append(tick)
                    else:
                        found = XT_OPERAND.search(raw)
                        if not found:
                            continue
                        entry = {"core": core, "engine": engine,
                                 "id": "m2:%d" % (int(found[1], 0) & 0xF),
                                 "pipe": (re.search(r"PIPE:(\w+)", raw) or [None, "?"])[1],
                                 "tick": tick, "released": None, "stall": None}
                        waits.append(entry)
                        pending.append(entry)
                elif kind in ("SET_FLAG", "WAIT_FLAG"):
                    found = FLAG.search(raw)
                    if found:
                        pipe, trigger, number = found.groups()
                        places["core%d.%s %s->%s id %d" % (
                            core, engine, pipe, trigger, int(number, 16))].append(
                            (tick, "SET" if kind.startswith("SET") else "WAIT"))
    return places, waits, dict(sets), seen


def audit_places(places):
    report, violations = {}, []
    for place, events in sorted(places.items()):
        events = sorted(events, key=lambda e: e[0])
        sequence = "".join("S" if kind == "SET" else "W" for _, kind in events)
        entry = {"sequence": sequence}
        held = 0
        for index, (tick, kind) in enumerate(events):
            if kind == "SET":
                held += 1
                if held > 1:
                    entry["fault"] = "double SET"
                    violations.append(
                        "%s: SET at tick %d while the place is still full (%s); a "
                        "HardEvent is one bit, so this token is lost"
                        % (place, tick, sequence[:index + 1]))
                    break
            else:
                held -= 1
                if held < 0:
                    entry["fault"] = "WAIT before SET"
                    violations.append(
                        "%s: WAIT at tick %d with the place empty (%s)"
                        % (place, tick, sequence[:index + 1]))
                    break
        else:
            if held:
                entry["fault"] = "token left"
                violations.append("%s: %d token(s) left at the end (%s)"
                                  % (place, held, sequence))
        report[place] = entry
    return report, violations


def audit_stalls(waits, sets, expect, expect_ids):
    """Per cross-core channel: what each wait on it actually stalled for.

    A channel whose FIRST wait stalls and whose later waits do not is the
    signature this exists for, so it is called out by name even without
    --expect-stall.
    """
    channels, violations = {}, []
    by_id = defaultdict(list)
    for entry in waits:
        by_id[(entry["core"], entry["engine"], entry["id"], entry["pipe"])].append(entry)
    for key, entries in sorted(by_id.items()):
        core, engine, flag, pipe = key
        entries.sort(key=lambda e: e["tick"])
        stalls = [e["stall"] for e in entries]
        # Every announcer of this channel except the consumer itself. For a
        # mode-2 quorum that is both AIVs; for a broadcast, the one AIC.
        producers = {who: sorted(ticks) for (chan, who), ticks in sets.items()
                     if chan == flag and who != engine}
        # The j-th increment of a consuming counter: for one producer its j-th
        # announcement, for a quorum the LATER of the producers' j-th ones.
        depth = min((len(t) for t in producers.values()), default=0)
        increments = [max(ticks[j] for ticks in producers.values())
                      for j in range(depth)]
        announced = sorted(t for ticks in producers.values() for t in ticks)
        name = "core%d.%s waits id %s on %s" % (core, engine, flag, pipe)
        channel = {"stalls": stalls, "wait_ticks": [e["tick"] for e in entries],
                   "producer_set_ticks": announced,
                   "counter_increment_ticks": increments}

        # Early retirement, measured rather than inferred from a stall floor.
        early = [(j, entries[j]["released"], increments[j])
                 for j in range(min(len(entries), len(increments)))
                 if entries[j]["released"] is not None
                 and entries[j]["released"] < increments[j]]
        if early:
            channel["fault"] = "early retirement"
            violations.append(
                "%s: wait %d was released at tick %d but the announcement that "
                "should have released it is at tick %d%s. A consuming counter "
                "releases the j-th wait with the j-th announcement; this wait "
                "did not block on one."
                % (name, early[0][0], early[0][1], early[0][2],
                   "" if len(early) == 1 else " (and %d more)" % (len(early) - 1)))
        known = [s for s in stalls if s is not None]
        # No scalar drain behind this wait, so every reading is the 1-2 tick
        # floor and this metric saw nothing. Say so rather than scoring it.
        if known and max(known) <= 2:
            channel["note"] = "no-fence: stall not measurable on this channel"
            channels[name] = channel
            continue
        if len(known) >= 2:
            first, rest = known[0], known[1:]
            spread = (max(increments) - min(increments)) / max(1, len(increments) - 1) \
                if len(increments) > 1 else 0
            # "Collapsed" = the first wait paid a real stall and the later ones
            # on the same id paid essentially nothing, while the producer's own
            # announcements are far apart. That is a wait that stopped blocking.
            if first > 1000 and max(rest) * 20 < first and spread > 1000:
                channel["fault"] = "stall collapsed after the first wait"
                violations.append(
                    "%s: stalls %s -- the first wait blocked for %d ticks and the "
                    "rest for at most %d, while this id's announcements are ~%d "
                    "ticks apart. The later waits were not released by an "
                    "announcement; this channel does not order its consumer."
                    % (name, known, first, max(rest), spread))
            # --expect-stall is scoped to the ids named with --expect-id,
            # because a stall floor is per channel and not global: the Cube's
            # per-strip kFlagScoreFree wait legitimately stalls ~1000 ticks on
            # one slot and ~19400 on the other, so one threshold over every
            # channel would fail a correct kernel.
            if (expect is not None and (not expect_ids or flag in expect_ids)
                    and any(s < expect for s in known)):
                channel.setdefault("fault", "stall below --expect-stall")
                violations.append("%s: stalls %s, expected every wait to stall at "
                                  "least %d ticks" % (name, known, expect))
        channels[name] = channel
    return channels, violations


# ----------------------------------------------------------------- self-test
# Synthetic dumps in the model's own format. One line per instruction, the tick
# in the second bracket, the engine before the colon, the mnemonic after the
# instruction id -- which is all `read()` parses.
def _line(tick, engine, index, text):
    return ("[info] [%08d] (PC: 0x1000) %-8s : (Binary: 0x0) (ID: %06d) %s\n"
            % (tick, engine, index, text))


def _trace(events):
    """events: [(tick, engine, mnemonic)] -> one dump file's text."""
    return "".join(_line(tick, engine, i, text)
                   for i, (tick, engine, text) in enumerate(events))


def _mode2_case(directory, aiv_release, aiv1_announce=200):
    """A one-channel mode-2 broadcast plus a two-producer quorum.

    The AIC announces the release at tick 1000. AIV0 waits and its drain --
    the SCALAR `WAIT_FLAG` behind the wait -- retires at `aiv_release`, so
    `aiv_release < 1000` is a wait that was not released by that announcement.
    The quorum channel's increment is the later of 100 and `aiv1_announce`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "core0.cubecore0.instr_popped_log.dump").write_text(_trace([
        (100, "FLOWCTRL", "WAIT_FLAG_DEV  PIPE:MTE1, XT:X0=0x8,"),
        (900, "SCALAR", "WAIT_FLAG  PIPE:MTE1, TRIGGER PIPE:SCALAR, FLAG ID:0, #V:0,"),
        (1000, "FLOWCTRL", "SET_CROSS_CORE  PIPE:FIXP, XT:X1=0x921,"),
    ]))
    (directory / "core0.veccore0.instr_popped_log.dump").write_text(_trace([
        (100, "FLOWCTRL", "SET_CROSS_CORE  PIPE:VEC, XT:X1=0x821,"),
        (110, "FLOWCTRL", "WAIT_FLAG_DEV  PIPE:VEC, XT:X0=0x9,"),
        (aiv_release, "SCALAR", "WAIT_FLAG  PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:1, #V:0,"),
    ]))
    (directory / "core0.veccore1.instr_popped_log.dump").write_text(_trace([
        (aiv1_announce, "FLOWCTRL", "SET_CROSS_CORE  PIPE:MTE3, XT:X1=0x821,"),
        (1100, "FLOWCTRL", "WAIT_FLAG_DEV  PIPE:MTE3, XT:X0=0x9,"),
        (1200, "SCALAR", "WAIT_FLAG  PIPE:MTE3, TRIGGER PIPE:SCALAR, FLAG ID:1, #V:0,"),
    ]))


def _mode4_case(directory, second_release):
    """Mode 4's collapse: two announcements far apart, the second wait early."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "core0.cubecore0.instr_popped_log.dump").write_text(_trace([
        (26337, "FLOWCTRL", "SET_INTRA_BLOCK  PIPE:FIXP, sync_id:26"),
        (47008, "FLOWCTRL", "SET_INTRA_BLOCK  PIPE:FIXP, sync_id:26"),
    ]))
    (directory / "core0.veccore1.instr_popped_log.dump").write_text(_trace([
        (3000, "FLOWCTRL", "WAIT_INTRA_BLOCK  PIPE:MTE3, sync_id:10"),
        (26340, "SCALAR", "WAIT_FLAG  PIPE:MTE3, TRIGGER PIPE:SCALAR, FLAG ID:0, #V:0,"),
        (26400, "FLOWCTRL", "WAIT_INTRA_BLOCK  PIPE:MTE3, sync_id:10"),
        (second_release, "SCALAR", "WAIT_FLAG  PIPE:MTE3, TRIGGER PIPE:SCALAR, FLAG ID:1, #V:0,"),
    ]))


def self_test():
    """The audit must reject what it exists to reject."""
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="tick-order-selftest-"))
    failures = 0
    try:
        cases = (
            # A release the consumer observed AFTER the announcement: clean.
            ("mode-2 release observed after the announcement",
             lambda d: _mode2_case(d, aiv_release=1400), False),
            # The same channel, released 600 ticks BEFORE the announcement.
            ("mode-2 release observed before the announcement",
             lambda d: _mode2_case(d, aiv_release=400), True),
            # The quorum rule: AIV1 announces at 2000, so the AIC's increment
            # is at 2000 and its drain retiring at 900 is early.
            ("mode-2 quorum released by one AIV alone",
             lambda d: _mode2_case(d, aiv_release=1400, aiv1_announce=2000), True),
            # Mode 4 as the operator measured it: the second wait retires at
            # 29614 while its announcement is at 47008.
            ("mode-4 second wait retiring before its announcement",
             lambda d: _mode4_case(d, second_release=29614), True),
            ("mode-4 second wait retiring after it",
             lambda d: _mode4_case(d, second_release=47100), False),
        )
        for index, (label, build, want_fail) in enumerate(cases):
            directory = root / ("case%d" % index)
            build(directory)
            places, waits, sets, files = read(directory, 0)
            _, violations = audit_stalls(waits, sets, None, set())
            got_fail = bool(violations)
            if got_fail != want_fail:
                print("  UNEXPECTED self-test %r: %s"
                      % (label, violations or "no violation"))
                failures += 1
            else:
                print("  %-52s %s" % (label, "rejected" if got_fail else "accepted"))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("%s: %d self-test failure(s)" % ("FAIL" if failures else "PASS", failures))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic controls instead of a trace; a "
                             "clean run of this audit is only evidence if it "
                             "can fail")
    parser.add_argument("--block", type=int,
                        help="audit only this physical block's three cores")
    parser.add_argument("--expect-stall", type=int, metavar="N",
                        help="fail any cross-core wait that stalled fewer than N "
                             "ticks; use on a rung where every wait must block")
    parser.add_argument("--expect-id", action="append", metavar="ID",
                        default=[],
                        help="restrict --expect-stall to this channel; repeatable. "
                             "A bare number is a mode-4 id, as the AIC addresses "
                             "it (AIV1's are rebased by 16); `m2:N` is a mode-2 "
                             "id, which is the same number on every core. Without "
                             "it the threshold applies to every measurable "
                             "channel, which is rarely what you want -- a stall "
                             "floor is per channel.")
    parser.add_argument("--cross", action="store_true",
                        help="print the per-channel stall census even when clean")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("early-retirement and stall-collapse self-test")
        return self_test()
    if args.directory is None:
        parser.error("EVIDENCE_DIR is required unless --self-test is given")

    places, waits, sets, files = read(args.directory, args.block)
    if not files or not places:
        print("no usable *.instr_popped_log.dump in %s" % args.directory, file=sys.stderr)
        return 2
    report, violations = audit_places(places)
    # A bare `--expect-id 26` still means mode 4, as it did before mode 2
    # existed; `m2:9` names a mode-2 channel.
    expect_ids = {name if name.startswith("m") else "m4:%s" % name
                  for name in args.expect_id}
    channels, stall_violations = audit_stalls(waits, sets, args.expect_stall,
                                              expect_ids)
    violations += stall_violations

    if args.json:
        print(json.dumps({"files": files, "intra_core_places": report,
                          "cross_core_channels": channels,
                          "violations": violations}, indent=2))
    else:
        print("%d completion logs, %d intra-core places, %d cross-core channels"
              % (files, len(report), len(channels)))
        for place, entry in report.items():
            print("  %-44s %-14s %s"
                  % (place, entry.get("fault", "ok"), entry["sequence"]))
        if args.cross or stall_violations:
            print("  -- cross-core stall census (ticks each wait blocked):")
            for name, channel in channels.items():
                mark = channel.get("fault") or channel.get("note") or "ok"
                print("  %-44s %-14s %s" % (name, mark, channel["stalls"]))
        for line in violations:
            print("VIOLATION: %s" % line)
    print("%s: %d violation(s) (%d intra-core order, %d cross-core stall)"
          % ("FAIL" if violations else "PASS", len(violations),
             len(violations) - len(stall_violations), len(stall_violations)))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
