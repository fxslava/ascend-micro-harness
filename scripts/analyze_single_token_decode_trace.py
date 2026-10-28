#!/usr/bin/env python3
"""Audit the AIV-only single-token decode operator from its CAModel traces.

Usage:
  analyze_single_token_decode_trace.py <aiv0_popped> <aiv0_complete>
                                       [--aiv1 <dump>] [--rows R]
                                       [--strips S] [--items I]

`--rows` is the rows per item, `--strips` the TOTAL strips the dump covers per
item (pages x strips-per-page), and `--items` the number of (request, kv_head)
items the audited core processed. Without them the quantitative checks report
instead of asserting. The model writes its dumps when the process stops, so a
runner that performs several launches leaves ONE dump covering all of them --
which is why the counts are totals and not per-launch.

Seven checks, each fatal. They are the pass criteria the numeric comparison
cannot express, and each catches a class the others are blind to.

  0. THE AUDITED CORE RAN. An idle core in a CAModel dump is not an empty one:
     it issues a scalar prologue and an early return, and every balance check
     below is vacuously true on it -- including one named "both Vector cores
     executed", whose test a single prologue MTE2 satisfies. So the core must
     first show the work it is defined by: vector arithmetic, MTE2 bursts and
     MTE3 bursts. This is also what catches auditing the wrong core, which cost
     the Cube decode operator six audited canary rungs.

  1. NO CROSS-CORE FLAG WAS ISSUED. This is the roadmap's gate 3.1.4, and it is
     a POSITIVE assertion about the design rather than an absence of findings.
     KERNEL_TYPE_AIV_ONLY with no Cube means there is nothing to hand off to,
     so SET_INTRA_BLOCK, WAIT_INTRA_BLOCK and every other mode-4 instruction
     must be ABSENT -- not balanced, absent. A balanced pair would still mean
     the kernel pays for a handshake it has no partner for, and a stale mode-4
     set is the mechanism behind two of the faults the Cube decode operator had
     to fix.

  2. NO CUBE INSTRUCTION WAS ISSUED, and no L1 traffic. The architectural
     isolation claim in full: no MMAD, no FIXPIPE, no MTE1 (the L1 -> L0 mover)
     and no L1 destination on any data mover. An AIV-only kernel that had
     quietly kept a CV transfer would still pass every numeric check, because
     the numbers do not depend on where the operand was staged.

  3. HARDEVENT BALANCE per (pipe, trigger pipe, flag id), so no token leaks
     across launches, and none of the reserved ids 6 or 7 is used. The route
     key comes from the instruction's OWN operand fields -- `PIPE:x, TRIGGER
     PIPE:y, FLAG ID:z` -- and not from the queue column it was dispatched on:
     a SET_FLAG is dispatched on its pipe and the matching WAIT_FLAG on the
     trigger pipe, so keying on the column pairs nothing with anything.

  3b. SET BEFORE WAIT in program order on every intra-core route. Instruction
     ids within one core ARE program order, and an intra-core SetFlag/WaitFlag
     pair is not a semaphore that waits for a set still to come: the set must
     be reached first. Counting cannot catch a violation -- the counts balance
     either way -- and the model's own diagnostic for it is "has no matched
     set_flag".

  4. THE SECOND VECTOR CORE DID NOTHING, in the isolated runs. With a one-block
     grid under KERNEL_TYPE_AIV_ONLY only one AIV may do work. If the companion
     dump shows vector arithmetic or an outflow, the launch was not the
     isolated single-item launch the canary protocol claims it was, and the
     protocol's "unassigned item slots remain unmutated" conclusion does not
     follow from it.

  5. THE EGRESS BUTTERFLY RAN EVERY STAGE ON EVERY ROW, as an exact count.
     Each lane-bit stage of the RegGather leaf is one `RV_VSELR` (the register
     lane permutation) plus one `RV_VMULA` (the fused partner + sign.self), and
     the Egress direction runs SIX lane-bit stages (strides 32, 1, 2, 4, 8, 16)
     over four registers per row -- strides 64 and 128 flip a register index
     bit and are straight-line Add/Sub with no permutation at all. So for R
     rows over I items:

         RV_VSELR == 6 stages x 4 registers x R x I

     and nothing else in this kernel issues a lane permutation, which is what
     makes that an equality rather than a bound.

     `RV_VMULA` is NOT the same number here, and the first version of this
     audit asserted that it was. It came back 288 against a predicted 144, and
     the trace says why: the extra instructions sit at six PCs that are not in
     the butterfly's code region, three per site, each retiring exactly once
     per 64-lane register, surrounded by an `RV_VMULS` against 0xbf800000 and
     an `RV_VABS_FP`. That is the 0-ULP FP32 `Div` -- THREE fused
     multiply-adds per register. This kernel divides twice in its epilogue
     (`acc / l`, one call per row over 4 registers each, and the sigmoid, one
     call over 4R registers), which is 8R register-iterations and so 24R
     multiply-adds:

         RV_VMULA == 24 x R x I  (butterfly)  +  24 x R x I  (the two divides)

     The row sweep is what makes that derivation falsifiable rather than
     fitted: both terms are linear in R, so --rows 1 and --rows 8 must land on
     48 and 384 and not on anything else.

     NOT asserted here: the butterfly's zero-UB-spill invariant, as an exact
     register load and store count. In this operator the level-2 continuous
     APIs lower to VF code too and dominate the register traffic, so a total
     over the dump cannot be attributed to the leaf. That invariant was
     measured where the butterfly WAS the whole kernel -- four loads and four
     stores per row and not one more, reports/rotate_butterfly.md -- and the
     totals here are reported for the census rather than gated.

  6. THE GLOBAL MEMORY TRAFFIC IS THE STRAIGHT LINE IT CLAIMS TO BE. Per item
     the kernel reads two 2 KiB packed strips and two scale slices per strip,
     plus one query panel and one gate panel; it writes one panel per strip and
     five epilogue bursts. Asserted as an exact load and store count, because a
     straight-line kernel's DMA count is a function of its trip count and
     nothing else -- a re-read would mean a buffer it thought it still had.

Then it reports the per-engine census and the per-strip interval, which is the
roadmap's gate 3.1.5: an instruction census and a per-engine cycle breakdown
against a budget registered before the run. Ticks come from one CAModel clock
in every dump, so they are directly comparable.

What this does NOT establish: silicon, a grid wider than the items launched, or
anything about the numbers -- that is the runner's bit-exact comparison.
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
# compare the sum of two routes against the sum of two others and the
# reserved-id check vacuous. Both alternatives, parsed with int(v, 0).
FLAG_ID_FIELD = re.compile(r"FLAG\s*ID\s*:\s*(0x[0-9a-fA-F]+|\d+)")
MOVER = re.compile(r"Src\s*:\s*(?P<src>\w+)\s*,\s*Dst\s*:\s*(?P<dst>\w+)")

CROSS_CORE = (
    "SET_INTRA_BLOCK",
    "WAIT_INTRA_BLOCK",
    "SET_INTER_BLOCK",
    "WAIT_INTER_BLOCK",
    "FFTS_CROSS_CORE_SYNC",
)
# Cube and L1: every mnemonic that would contradict the AIV-only claim.
CUBE_MNEMONICS = ("MMAD", "FIXPIPE", "LOAD_L0", "LOAD2D", "LOAD3D", "BROAD_CAST")
CUBE_ENGINES = ("MTE1", "M", "FIX", "CUBE")
LOAD_MNEMONICS = {"RV_VLD", "RV_VLDA", "RV_VLDU", "RV_VLDI"}
STORE_MNEMONICS = {"RV_VST", "RV_VSTA", "RV_VSTU", "RV_VSTI"}
# The Egress direction's lane-bit stages: strides 32, 1, 2, 4, 8, 16. Strides
# 64 and 128 flip a register index bit and issue no permutation.
EGRESS_LANE_STAGES = 6
ROT_REGISTERS = 4
# Fused multiply-adds the 0-ULP FP32 `Div` issues per 64-lane register, read
# off the trace: three distinct PCs per divide site, each retiring once per
# register. See check 5.
DIV_FMA_PER_REGISTER = 3

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
    """The data movers, with direction. `OUT` is Global Memory."""
    out = []
    for row in rows:
        if not row["mnemonic"].startswith("MOV_SRC_TO_DST"):
            continue
        match = MOVER.search(row["tail"])
        if not match:
            continue
        out.append(
            dict(id=row["id"], tick=row["tick"], engine=row["engine"],
                 src=match.group("src"), dst=match.group("dst"))
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("popped")
    parser.add_argument("completed")
    parser.add_argument("--aiv1", default="")
    parser.add_argument("--rows", type=int, default=0,
                        help="rows per item")
    parser.add_argument("--strips", type=int, default=0,
                        help="total strips per item across every page in this dump")
    parser.add_argument("--items", type=int, default=0,
                        help="(request, kv_head) items the audited core processed")
    args = parser.parse_args()

    popped = parse(args.popped)
    if not popped:
        print(f"FAIL no instruction records in {args.popped}")
        return 1
    completed = parse(args.completed)
    counts = Counter(row["mnemonic"] for row in popped)
    engines = Counter(row["engine"] for row in popped)

    # In the concurrent-grid run the audited core takes ONE item of the grid,
    # not all of them: the kernel's stride loop is `item += block_num`, so with
    # block_num == items each core processes exactly one. The per-core
    # expectation is therefore one item whatever --items says the launch held.
    items_on_core = 1 if args.items > 0 else 0

    # ---- 0. the audited core ran ---------------------------------------
    vector = sum(v for k, v in engines.items() if k.startswith("VEC") or k.startswith("RVEC"))
    expect(vector > 0, "check 0: the audited core issued vector instructions")
    expect(engines.get("MTE2", 0) > 0, "check 0: the audited core issued MTE2 bursts")
    expect(engines.get("MTE3", 0) > 0, "check 0: the audited core issued MTE3 bursts")

    # ---- 1. no cross-core flag ------------------------------------------
    cross = {m: counts[m] for m in CROSS_CORE if counts[m]}
    expect(not cross,
           f"check 1: no cross-core flag instruction was issued ({cross or 'none'})")

    # ---- 2. no Cube, no L1 ----------------------------------------------
    cube = {m: c for m, c in counts.items()
            if any(m.startswith(prefix) for prefix in CUBE_MNEMONICS)}
    expect(not cube, f"check 2: no Cube instruction was issued ({cube or 'none'})")
    cube_engine = {e: c for e, c in engines.items() if e in CUBE_ENGINES}
    expect(not cube_engine,
           f"check 2: no Cube or MTE1 engine appears in the dump ({cube_engine or 'none'})")
    l1_traffic = [m for m in movers(popped) if "L1" in (m["src"], m["dst"])]
    expect(not l1_traffic, f"check 2: no data mover touched L1 ({len(l1_traffic)})")

    # ---- 3. HardEvent balance -------------------------------------------
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
    expect(not leaked,
           f"check 3: SET_FLAG == WAIT_FLAG per (pipe, trigger, id) ({leaked})")
    print(f"     routes: {sorted((k, len(sets[k])) for k in set(sets) | set(waits))}")

    reserved = [key for key in set(sets) | set(waits) if key[2] in (6, 7)]
    expect(not reserved, f"check 3: no reserved flag id 6 or 7 is used ({reserved})")

    # ---- 3b. set before wait --------------------------------------------
    offenders = []
    for key in sorted(set(sets) & set(waits)):
        s = sorted(sets[key])
        w = sorted(waits[key])
        for i in range(min(len(s), len(w))):
            if s[i] > w[i]:
                offenders.append((key, s[i], w[i]))
                break
    expect(not offenders,
           f"check 3b: every intra-core SET precedes its WAIT in program order ({offenders})")

    # ---- 4. the second Vector core did nothing --------------------------
    if args.aiv1 and Path(args.aiv1).exists():
        other = parse(args.aiv1)
        other_engines = Counter(row["engine"] for row in other)
        other_vec = sum(v for k, v in other_engines.items()
                        if k.startswith("VEC") or k.startswith("RVEC"))
        other_stores = sum(1 for m in movers(other) if m["dst"] == "OUT")
        if args.items and args.items > 1:
            print(f"ok   check 4: skipped, the grid holds {args.items} items so the "
                  f"companion core is expected to work ({other_vec} vector instr)")
        else:
            expect(other_vec == 0,
                   f"check 4: the companion Vector core issued no vector work ({other_vec})")
            expect(other_stores == 0,
                   f"check 4: the companion Vector core issued no outflow ({other_stores})")
        print(f"     companion core census: {dict(other_engines) or 'empty'}")
    else:
        print("ok   check 4: skipped, no companion dump given")

    # ---- 5. the egress butterfly did not spill --------------------------
    vselr = counts["RV_VSELR"]
    vmula = counts["RV_VMULA"]
    rv_vld = sum(counts[m] for m in LOAD_MNEMONICS)
    rv_vst = sum(counts[m] for m in STORE_MNEMONICS)
    if args.rows > 0 and items_on_core > 0:
        butterfly = EGRESS_LANE_STAGES * ROT_REGISTERS * args.rows * items_on_core
        # Two 0-ULP divides in the epilogue: `acc / l` is one call per row over
        # ROT_REGISTERS registers, the sigmoid one call over ROT_REGISTERS x R.
        div_registers = 2 * ROT_REGISTERS * args.rows * items_on_core
        want_vmula = butterfly + DIV_FMA_PER_REGISTER * div_registers
        expect(vselr == butterfly,
               f"check 5: RV_VSELR {vselr} == {EGRESS_LANE_STAGES} stages x "
               f"{ROT_REGISTERS} regs x {args.rows} rows x {items_on_core} item(s) "
               f"= {butterfly}")
        expect(vmula == want_vmula,
               f"check 5: RV_VMULA {vmula} == {butterfly} butterfly + "
               f"{DIV_FMA_PER_REGISTER} x {div_registers} divide registers "
               f"= {want_vmula}")
        print(f"     register traffic, reported not gated: RV_VLD {rv_vld}, "
              f"RV_VST {rv_vst} (the level-2 APIs lower to VF code too, so this "
              "total is not attributable to the butterfly leaf)")
    else:
        print(f"ok   check 5: reported only -- RV_VSELR {vselr}, RV_VMULA {vmula}, "
              f"RV_VLD {rv_vld}, RV_VST {rv_vst} (pass --rows/--items to assert)")

    # ---- 6. the Global Memory traffic ------------------------------------
    moved = movers(popped)
    loads = [m for m in moved if m["src"] == "OUT"]
    stores = [m for m in moved if m["dst"] == "OUT"]
    expect(bool(loads) and bool(stores),
           "check 6: the core both read and wrote Global Memory")
    if args.strips > 0 and args.rows > 0 and items_on_core > 0:
        # Per launch: one pi_signs burst. Per item: query panel + gate panel.
        # Per strip: packed key, packed value, key scale, value scale.
        want_loads = 1 + items_on_core * (2 + 4 * args.strips)
        # Per item: one panel archive per strip, plus m, l, acc, rotated,
        # context. Plus the launch probe, written once by block 0.
        want_stores = 1 + items_on_core * (args.strips + 5)
        expect(len(loads) == want_loads,
               f"check 6: {len(loads)} GM loads == 1 pi_signs + {items_on_core} x "
               f"(2 panels + 4 x {args.strips} strip operands) = {want_loads}")
        expect(len(stores) == want_stores,
               f"check 6: {len(stores)} GM stores == 1 probe + {items_on_core} x "
               f"({args.strips} archived panels + 5 epilogue bursts) = {want_stores}")
    else:
        print(f"     reported only: {len(loads)} GM->UB loads, {len(stores)} UB->GM "
              "stores (pass --strips/--rows/--items to assert)")

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
    print("vector and DMA mnemonic census (dispatch):")
    for mnemonic, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if (mnemonic.startswith("RV_") or mnemonic.startswith("V") or
                mnemonic.startswith("MOV_SRC") or
                mnemonic in ("SET_FLAG", "WAIT_FLAG", "VF")):
            print(f"  {mnemonic:<20} {count:>6}")
    all_ticks = [row["tick"] for row in popped]
    span = max(all_ticks) - min(all_ticks)
    print()
    print(f"whole-dump dispatch span: {span} model ticks -- fill plus drain over "
          "every launch, NOT a throughput figure")
    if args.strips > 0 and items_on_core > 0:
        per_strip = span / float(args.strips * items_on_core)
        print(f"per-strip dispatch interval: {per_strip:.0f} ticks over "
              f"{args.strips * items_on_core} strip(s), epilogue included")
    if completed:
        print(f"completed-log instructions: {len(completed)}")

    if failures:
        print()
        print(f"{len(failures)} audit check(s) failed")
        return 1
    print()
    print("single-token decode trace audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
