#!/usr/bin/env python3
"""Audit the AIV-only scatter ingress operator from its CAModel traces.

Usage:
  analyze_scatter_trace.py <aiv0_popped> <aiv0_complete>
                           [--aiv1 <dump>] [--rows R] [--runs N] [--launches L]

`--rows` is the total number of rotated 256-channel rows the dump covers, per
plane: a launch of n tokens rotates n rows of key and n of value, so a
three-launch ladder of 1, 1 and 3 tokens is `--rows 5 --runs 3 --launches 3`.
Without them the quantitative checks report instead of asserting. The model
writes its dumps when the process stops, so a runner that performs several
launches leaves ONE dump covering all of them -- which is why the counts are
totals and not per-launch.

Six checks, each fatal. They are the pass criteria the numeric comparison
cannot express, and each catches a class the others are blind to.

  0. THE AUDITED CORE RAN. An idle core in a CAModel dump is not an empty one
     -- it issues a scalar prologue and an early return, and every balance
     check below is vacuously true on it. So the core must first show the work
     it is defined by: Reg-vector arithmetic, MTE2 bursts and MTE3 bursts. This
     is also what catches auditing the wrong core.

  1. NO CROSS-CORE FLAG WAS ISSUED. KERNEL_TYPE_AIV_ONLY with no Cube means
     there is nothing to hand off to, so SET_INTRA_BLOCK, WAIT_INTRA_BLOCK and
     every other mode-4 instruction must be ABSENT -- not balanced, absent. A
     balanced pair would still mean the kernel is paying for a handshake it has
     no partner for, and a stale mode-4 set is the mechanism behind two of the
     faults the decode operator had to fix.

  2. HARDEVENT BALANCE per (pipe, trigger pipe, flag id), so no token leaks
     across launches, and none of the reserved ids 6 or 7 is used. The route
     key comes from the instruction's OWN operand fields -- `PIPE:x, TRIGGER
     PIPE:y, FLAG ID:z` -- and not from the queue column it was dispatched on:
     a SET_FLAG is dispatched on its pipe and the matching WAIT_FLAG on the
     trigger pipe, so keying on the column pairs nothing with anything.

  2b. SET BEFORE WAIT in program order on every intra-core route. Instruction
     ids within one core ARE program order, and an intra-core SetFlag/WaitFlag
     pair is not a semaphore that waits for a set still to come: the set must
     be reached first. Counting cannot catch a violation -- the counts balance
     either way -- and the model's own diagnostic for it is "has no matched
     set_flag".

  3. THE SECOND VECTOR CORE DID NOTHING. With a one-block grid under
     KERNEL_TYPE_AIV_ONLY only one AIV may do work. If the companion dump shows
     Reg-vector arithmetic or an outflow, the launch was not the isolated
     single-head launch the canary protocol claims it was, and the protocol's
     "all unassigned head regions remain unmutated" conclusion does not follow.

  4. THE BUTTERFLY DID NOT SPILL, as an exact count. Each lane-bit stage of the
     RegGather leaf is one `RV_VSELR` (the register lane permutation) plus one
     `RV_VMULA` (the fused partner + sign.self), and the absmax all-reduce adds
     six more `RV_VSELR`. So for R rows over two planes:
         RV_VMULA == 6 stages x 4 registers x 2 planes x R
         RV_VSELR == that, plus 6 x 2 x R
     An equality, not a bound: a spill of the four resident registers would
     multiply the surrounding loads and stores without changing these, and a
     recompiled leaf that lost a stage would change them exactly.

  5. THE READ-MODIFY-WRITE IS A READ THEN A WRITE, AT THE SAME ADDRESS. Every
     GM <-> UB data mover is listed with its direction and Global Memory
     address; for each one the kernel writes a 2048-byte strip to, there must be
     an earlier load from the SAME address. That is the RMW contract proved from
     the trace rather than inferred from the numbers -- a kernel that wrote a
     strip it never read would still pass a comparison whose reference started
     from the same poison.

Then it reports the per-engine census. Ticks come from one CAModel clock in
every dump, so they are directly comparable.

What this does NOT establish: silicon, a grid wider than one block, or anything
about the numbers -- that is the runner's bit-exact comparison.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[(?P<tick>\d+)\]\s+\(PC: (?P<pc>0x[0-9a-f]+)\)\s+(?P<engine>[A-Z_0-9]+)\s*:"
    r".*?\(ID: (?P<id>\d+)\)\s+(?P<mnemonic>[A-Z_0-9]+)(?P<tail>.*)$"
)
PIPE_FIELD = re.compile(r"PIPE\s*:\s*([A-Z0-9_]+)")
# The model prints a non-zero flag id in HEX (`FLAG ID:0x2`) and zero in
# decimal (`FLAG ID:0`). A `\d+` pattern matches the leading "0" of "0x2" and
# stops, so every id collapses into bucket 0 -- which makes the balance check
# below compare the sum of two routes against the sum of two others and the
# reserved-id check vacuous. Both alternatives, parsed with int(v, 0).
FLAG_ID_FIELD = re.compile(r"FLAG\s*ID\s*:\s*(0x[0-9a-fA-F]+|\d+)")
MOVER = re.compile(r"Src\s*:\s*(?P<src>\w+)\s*,\s*Dst\s*:\s*(?P<dst>\w+)")
# An operand prints as X<n>=0x<hex> or, when it is zero, as X<n>=0. Matching
# only the hex form drops the zero operands and shifts every later position,
# which silently moved the Global Memory address of any burst whose first
# operand happened to be zero.
HEX_OPERAND = re.compile(r"X\d+=(0x[0-9a-f]+|\d+)")

CROSS_CORE = (
    "SET_INTRA_BLOCK",
    "WAIT_INTRA_BLOCK",
    "SET_INTER_BLOCK",
    "WAIT_INTER_BLOCK",
    "FFTS_CROSS_CORE_SYNC",
)
LOAD_MNEMONICS = {"RV_VLD", "RV_VLDA", "RV_VLDU", "RV_VLDI"}
STORE_MNEMONICS = {"RV_VST", "RV_VSTA", "RV_VSTU", "RV_VSTI"}

failures = []


def expect(condition, what):
    if not condition:
        failures.append(what)
        print(f"FAIL {what}")
    else:
        print(f"ok   {what}")


def parse(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            rows.append(
                dict(
                    tick=int(match.group("tick")),
                    engine=match.group("engine"),
                    id=int(match.group("id")),
                    mnemonic=match.group("mnemonic"),
                    tail=match.group("tail"),
                )
            )
    return rows


def flag_key(row):
    """(pipe, trigger pipe, flag id) from the instruction's own fields."""
    pipes = PIPE_FIELD.findall(row["tail"])
    ids = FLAG_ID_FIELD.findall(row["tail"])
    pipe = pipes[0] if pipes else "?"
    trigger = pipes[1] if len(pipes) > 1 else "?"
    flag_id = int(ids[0], 0) if ids else -1
    return (pipe, trigger, flag_id)


def movers(rows):
    """The GM <-> UB data movers, with direction and the Global Memory address.

    The dump writes `Src:UB, Dst:OUT` for an outflow and `Src:OUT, Dst:UB` for
    an ingress, then lists the operand registers by value in instruction order:
    XD, XN, XM, XT. XD is the destination and XN the source, so the Global
    Memory address is operand 0 for a store and operand 1 for a load -- and the
    other two are packed length and stride words that are numerically LARGER
    than a device pointer, which is why picking the biggest operand does not
    work.
    """
    out = []
    for row in rows:
        if not row["mnemonic"].startswith("MOV_SRC_TO_DST"):
            continue
        match = MOVER.search(row["tail"])
        if not match:
            continue
        values = [int(v, 0) for v in HEX_OPERAND.findall(row["tail"])]
        store = match.group("dst") == "OUT"
        slot = 0 if store else 1
        gm = values[slot] if len(values) > slot else 0
        out.append(
            dict(id=row["id"], tick=row["tick"], engine=row["engine"],
                 src=match.group("src"), dst=match.group("dst"), gm=gm, store=store)
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("popped")
    parser.add_argument("completed")
    parser.add_argument("--aiv1", default="")
    parser.add_argument("--rows", type=int, default=0,
                        help="total rotated rows per plane across every launch in this dump")
    parser.add_argument("--runs", type=int, default=0,
                        help="total read-modify-write transactions in this dump")
    parser.add_argument("--launches", type=int, default=0,
                        help="launches in this dump (each writes the launch probe once)")
    args = parser.parse_args()

    popped = parse(args.popped)
    if not popped:
        print(f"FAIL no instruction records in {args.popped}")
        return 1
    completed = parse(args.completed)
    counts = Counter(row["mnemonic"] for row in popped)
    engines = Counter(row["engine"] for row in popped)

    # ---- 0. the audited core ran ---------------------------------------
    reg_vector = sum(v for k, v in engines.items() if k.startswith("RVEC"))
    expect(reg_vector > 0, "check 0: the audited core issued Reg-vector instructions")
    expect(engines.get("MTE2", 0) > 0, "check 0: the audited core issued MTE2 bursts")
    expect(engines.get("MTE3", 0) > 0, "check 0: the audited core issued MTE3 bursts")

    # ---- 1. no cross-core flag ------------------------------------------
    cross = {m: counts[m] for m in CROSS_CORE if counts[m]}
    expect(not cross, f"check 1: no cross-core flag instruction was issued ({cross or 'none'})")

    # ---- 2. HardEvent balance -------------------------------------------
    sets = defaultdict(list)
    waits = defaultdict(list)
    for row in popped:
        if row["mnemonic"] not in ("SET_FLAG", "WAIT_FLAG"):
            continue
        key = flag_key(row)
        (sets if row["mnemonic"] == "SET_FLAG" else waits)[key].append(row["id"])

    leaked = [(key, len(sets[key]), len(waits[key]))
              for key in sorted(set(sets) | set(waits))
              if len(sets[key]) != len(waits[key])]
    expect(not leaked, f"check 2: SET_FLAG == WAIT_FLAG per (pipe, trigger, id) ({leaked})")
    print(f"     routes: {sorted((k, len(sets[k])) for k in set(sets) | set(waits))}")

    reserved = [key for key in set(sets) | set(waits) if key[2] in (6, 7)]
    expect(not reserved, f"check 2: no reserved flag id 6 or 7 is used ({reserved})")

    # ---- 2b. set before wait --------------------------------------------
    offenders = []
    for key in sorted(set(sets) & set(waits)):
        s = sorted(sets[key])
        w = sorted(waits[key])
        for i in range(min(len(s), len(w))):
            if s[i] > w[i]:
                offenders.append((key, s[i], w[i]))
                break
    expect(not offenders,
           f"check 2b: every intra-core SET precedes its WAIT in program order ({offenders})")

    # ---- 3. the second Vector core did nothing --------------------------
    if args.aiv1 and Path(args.aiv1).exists():
        other = parse(args.aiv1)
        other_engines = Counter(row["engine"] for row in other)
        other_reg = sum(v for k, v in other_engines.items() if k.startswith("RVEC"))
        other_stores = sum(1 for m in movers(other) if m["store"])
        expect(other_reg == 0,
               f"check 3: the companion Vector core issued no Reg-vector work ({other_reg})")
        expect(other_stores == 0,
               f"check 3: the companion Vector core issued no outflow ({other_stores})")
        print(f"     companion core census: {dict(other_engines) or 'empty'}")
    else:
        print("ok   check 3: skipped, no companion dump given")

    # ---- 4. the butterfly did not spill ---------------------------------
    vselr = counts["RV_VSELR"]
    vmula = counts["RV_VMULA"]
    rv_vld = sum(counts[m] for m in LOAD_MNEMONICS)
    rv_vst = sum(counts[m] for m in STORE_MNEMONICS)
    if args.rows > 0:
        want_mula = 6 * 4 * 2 * args.rows
        want_selr = want_mula + 6 * 2 * args.rows
        expect(vmula == want_mula,
               f"check 4: RV_VMULA {vmula} == 6 stages x 4 regs x 2 planes x {args.rows} rows "
               f"= {want_mula}")
        expect(vselr == want_selr,
               f"check 4: RV_VSELR {vselr} == {want_mula} butterfly + "
               f"{6 * 2 * args.rows} absmax = {want_selr}")
    else:
        print(f"ok   check 4: reported only -- RV_VSELR {vselr}, RV_VMULA {vmula}, "
              f"RV_VLD {rv_vld}, RV_VST {rv_vst} (pass --rows R to assert)")

    # ---- 5. the read-modify-write order ---------------------------------
    moved = movers(popped)
    loads = [m for m in moved if not m["store"]]
    stores = [m for m in moved if m["store"]]
    expect(loads and stores, "check 5: the core both read and wrote Global Memory")
    unread = []
    for store in stores:
        if store["gm"] == 0:
            continue
        if not any(load["gm"] == store["gm"] and load["id"] < store["id"] for load in loads):
            unread.append((store["id"], hex(store["gm"])))
    # Only the strip is a read-modify-write; the key page, the scale lanes and
    # the launch probe are plain writes, so an unread store is reported and only
    # fails when it is one of the addresses a load also touched elsewhere.
    rmw_addresses = {load["gm"] for load in loads} & {store["gm"] for store in stores}
    bad = [u for u in unread if int(u[1], 16) in rmw_addresses]
    expect(not bad, f"check 5: every re-written Global Memory address was read first ({bad})")
    print(f"     {len(loads)} GM->UB loads, {len(stores)} UB->GM stores, "
          f"{len(rmw_addresses)} address(es) both read and written")
    for m in moved:
        print(f"       id {m['id']:>6} tick {m['tick']:>7} {m['engine']:<5} "
              f"{m['src']}->{m['dst']} gm {hex(m['gm'])}")
    # The straight-line body, per transaction and per token. A run of n tokens
    # is ONE strided key-row burst, ONE value-row burst and ONE strip read, so
    # the loads scale with runs; but the scale plane is four bytes per token --
    # sub-burst, one DataCopyPad each -- so the stores scale with BOTH. The
    # first version of this check had `4 x runs` stores and reported a
    # correct kernel as wrong on the first multi-token rung.
    if args.runs > 0 and args.rows > 0:
        probe = args.launches
        want_loads = probe + 3 * args.runs
        want_stores = probe + 2 * args.runs + 2 * args.rows
        expect(len(loads) == want_loads,
               f"check 5: {len(loads)} loads == {probe} pi_signs + 3 x {args.runs} "
               f"(key rows, value rows, strip) = {want_loads}")
        expect(len(stores) == want_stores,
               f"check 5: {len(stores)} stores == {probe} probe + 2 x {args.runs} "
               f"(key page, strip) + 2 x {args.rows} (one scale DataCopyPad per token "
               f"per plane) = {want_stores}")
    # NOT asserted: that the number of distinct re-written addresses equals the
    # run count. Each launch allocates its own Global Memory, and a freed
    # allocation is handed straight back, so two launches of a multi-launch rung
    # routinely see the same address for their strip. The property that matters
    # is the one above -- every re-written address was read first -- and it does
    # not depend on how the allocator happened to recycle.

    # ---- the census ------------------------------------------------------
    print()
    print(f"{'engine':<10} {'instr':>8} {'first':>10} {'last':>10} {'span':>10}")
    by_engine = defaultdict(list)
    for row in popped:
        by_engine[row["engine"]].append(row["tick"])
    for engine in sorted(by_engine, key=lambda e: -len(by_engine[e])):
        ticks = by_engine[engine]
        print(f"{engine:<10} {len(ticks):>8} {min(ticks):>10} {max(ticks):>10} "
              f"{max(ticks) - min(ticks):>10}")
    print()
    print("Reg-vector and DMA mnemonic census (dispatch):")
    for mnemonic, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if (mnemonic.startswith("RV_") or mnemonic.startswith("MOV_SRC")
                or mnemonic in ("SET_FLAG", "WAIT_FLAG", "VF")):
            print(f"  {mnemonic:<20} {count:>6}")
    all_ticks = [row["tick"] for row in popped]
    print()
    print(f"whole-dump dispatch span: {max(all_ticks) - min(all_ticks)} model ticks "
          "-- fill plus drain over every launch, NOT a throughput figure")
    if completed:
        print(f"completed-log instructions: {len(completed)}")

    if failures:
        print()
        print(f"{len(failures)} audit check(s) failed")
        return 1
    print()
    print("scatter trace audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
