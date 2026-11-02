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
   fenced kernel places right behind the wait. A mode-4 wait blocks only the
   issuance of its own pipe (`CrossCoreWaitFlag` on Ascend 950PR), so the
   `SetFlag<x_S>` / `WaitFlag<x_S>` pair behind it stalls instruction issue
   until pipe x has retired past the wait -- and that `WaitFlag` is a SCALAR
   instruction whose logged tick IS when it ran. The gap from the wait to the
   next scalar instruction is therefore the stall the wait actually imposed.

   This is worth a gate because a wait that silently stops blocking looks
   exactly like a wait that had nothing to wait for. On the four-page rung
   AIV1's four waits on `kFlagPageDrainedV1`, a
   `CrossCoreWaitFlag<4, PIPE_MTE3>`, stalled 23146, 3, 3 and 3 ticks while
   the Cube's four announcements of that id are ~20600 ticks apart.
   `--expect-stall N` fails a channel whose later waits collapse like that,
   which is the regression test for it.

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

usage: audit_mode4_tick_order.py EVIDENCE_DIR [--block N] [--expect-stall N]
                                 [--cross] [--json]
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
CORE = re.compile(r"core(\d+)\.(\w+)\.instr_popped_log\.dump")


def read(directory, block):
    """-> (places, waits, files).

    places: {place: [(tick, kind)]} for the intra-core HardEvent routes.
    waits:  [{core, engine, id, tick, released, stall}] per cross-core wait,
            where `released` is the next SCALAR instruction's tick.
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
                # A blocked wait shows up as a gap before the next SCALAR
                # instruction, which is the x_S drain behind it.
                if pending and unit == "SCALAR":
                    for entry in pending:
                        entry["released"] = tick
                        entry["stall"] = tick - entry["tick"]
                    pending = []
                if kind in ("SET_INTRA_BLOCK", "WAIT_INTRA_BLOCK"):
                    found = MODE4.search(raw)
                    if not found:
                        continue
                    flag = int(found[1]) + rebase
                    if kind == "SET_INTRA_BLOCK":
                        sets[flag].append(tick)
                    else:
                        entry = {"core": core, "engine": engine, "id": flag,
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
        announced = sets.get(flag, [])
        name = "core%d.%s waits id %d on %s" % (core, engine, flag, pipe)
        channel = {"stalls": stalls, "wait_ticks": [e["tick"] for e in entries],
                   "producer_set_ticks": announced}
        known = [s for s in stalls if s is not None]
        # No scalar drain behind this wait, so every reading is the 1-2 tick
        # floor and this metric saw nothing. Say so rather than scoring it.
        if known and max(known) <= 2:
            channel["note"] = "no-fence: stall not measurable on this channel"
            channels[name] = channel
            continue
        if len(known) >= 2:
            first, rest = known[0], known[1:]
            spread = (max(announced) - min(announced)) / max(1, len(announced) - 1) \
                if len(announced) > 1 else 0
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--block", type=int,
                        help="audit only this physical block's three cores")
    parser.add_argument("--expect-stall", type=int, metavar="N",
                        help="fail any cross-core wait that stalled fewer than N "
                             "ticks; use on a rung where every wait must block")
    parser.add_argument("--expect-id", type=int, action="append", metavar="ID",
                        default=[],
                        help="restrict --expect-stall to this absolute mode-4 id; "
                             "repeatable. Without it the threshold applies to "
                             "every measurable channel, which is rarely what you "
                             "want -- a stall floor is per channel.")
    parser.add_argument("--cross", action="store_true",
                        help="print the per-channel stall census even when clean")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    places, waits, sets, files = read(args.directory, args.block)
    if not files or not places:
        print("no usable *.instr_popped_log.dump in %s" % args.directory, file=sys.stderr)
        return 2
    report, violations = audit_places(places)
    channels, stall_violations = audit_stalls(waits, sets, args.expect_stall,
                                              set(args.expect_id))
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
